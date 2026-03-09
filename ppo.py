"""
Standalone PPO (Proximal Policy Optimization) agent for evaluating robot morphologies.
Implemented from scratch using PyTorch — no dependency on the repo's algo/ or tasks/ code.

Usage:
    python ppo.py --xml_path assets/base_ant_flat.xml --total_steps 500000
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv as GymAntEnv


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk MLP with separate heads for policy (actor) and value (critic).
    Actor outputs mean of a Gaussian; log_std is a learned parameter vector.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )

        # Policy head — outputs action mean
        self.actor_mean = nn.Linear(hidden, act_dim)
        # Log std as a learnable parameter (independent of obs)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Value head
        self.critic = nn.Linear(hidden, 1)

        # Orthogonal init (standard for MuJoCo PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        value = self.critic(features).squeeze(-1)
        return dist, value

    def act(self, obs: torch.Tensor):
        """Sample action, return (action, log_prob, value)."""
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)  # sum over action dims
        return action, log_prob, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate stored actions under current policy."""
        dist, value = self(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores one batch of experience collected from the environment."""

    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, device: torch.device):
        self.n_steps = n_steps
        self.device = device

        self.obs      = torch.zeros(n_steps, obs_dim,  device=device)
        self.actions  = torch.zeros(n_steps, act_dim,  device=device)
        self.log_probs= torch.zeros(n_steps,           device=device)
        self.rewards  = torch.zeros(n_steps,           device=device)
        self.values   = torch.zeros(n_steps,           device=device)
        self.dones    = torch.zeros(n_steps,           device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.values[self.ptr]    = value
        self.dones[self.ptr]     = done
        self.ptr += 1

    def compute_returns(self, last_value: float, gamma: float, lam: float):
        """GAE-λ advantage estimation."""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t].item()
            next_value = self.values[t + 1].item() if t + 1 < self.n_steps else last_value
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta.item() + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        self.returns = advantages + self.values
        self.advantages = advantages

    def get_minibatches(self, n_minibatches: int):
        """Yield shuffled minibatches of (obs, actions, log_probs, returns, advantages)."""
        indices = torch.randperm(self.n_steps, device=self.device)
        mb_size = self.n_steps // n_minibatches
        # Normalize advantages
        adv = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        for start in range(0, self.n_steps, mb_size):
            idx = indices[start:start + mb_size]
            yield (
                self.obs[idx],
                self.actions[idx],
                self.log_probs[idx],
                self.returns[idx],
                adv[idx],
            )

    def reset(self):
        self.ptr = 0


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float          = 3e-4,
        gamma: float       = 0.99,
        lam: float         = 0.95,
        clip_eps: float    = 0.2,
        n_steps: int       = 2048,
        n_minibatches: int = 32,
        n_epochs: int      = 10,
        vf_coef: float     = 0.5,
        ent_coef: float    = 0.0,
        max_grad_norm: float = 0.5,
        device: str        = "cpu",
    ):
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.n_steps      = n_steps
        self.n_minibatches= n_minibatches
        self.n_epochs     = n_epochs
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.max_grad_norm= max_grad_norm
        self.device       = torch.device(device)

        self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(n_steps, obs_dim, act_dim, self.device)

    @torch.no_grad()
    def collect_rollout(self, env, obs: np.ndarray):
        """Fill the rollout buffer with n_steps of experience."""
        self.buffer.reset()
        ep_rewards, ep_lengths = [], []
        ep_reward, ep_length = 0.0, 0

        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, value = self.policy.act(obs_t)

            # Clamp action to env limits
            act_np = action.cpu().numpy()
            act_np = np.clip(act_np, env.action_space.low, env.action_space.high)

            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            self.buffer.add(obs_t, action, log_prob, 
                            torch.tensor(reward, device=self.device),
                            value,
                            torch.tensor(float(done), device=self.device))

            obs = next_obs
            ep_reward += reward
            ep_length += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward, ep_length = 0.0, 0
                obs, _ = env.reset()

        # Bootstrap value for last step
        last_obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        _, last_value = self.policy(last_obs_t)
        self.buffer.compute_returns(last_value.item(), self.gamma, self.lam)

        return obs, ep_rewards, ep_lengths

    def update(self):
        """Run PPO epochs over the collected rollout."""
        total_pg_loss = total_vf_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, returns, advantages in \
                    self.buffer.get_minibatches(self.n_minibatches):

                log_probs, values, entropy = self.policy.evaluate(obs, actions)

                # Ratio for clipped PPO objective
                ratio = (log_probs - old_log_probs).exp()
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value function loss (clipped)
                vf_loss = 0.5 * (returns - values).pow(2).mean()

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent     += (-ent_loss).item()
                n_updates += 1

        return {
            "pg_loss": total_pg_loss / n_updates,
            "vf_loss": total_vf_loss / n_updates,
            "entropy": total_ent     / n_updates,
        }

    def evaluate_policy(self, env, n_episodes: int = 5) -> float:
        """
        Run n_episodes with the current (deterministic) policy and return
        mean episode return. Used as the fitness score for a robot morphology.
        """
        returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    dist, _ = self.policy(obs_t)
                action = dist.mean.cpu().numpy()  # deterministic: use mean
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)
        return float(np.mean(returns))

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Build env using the custom AntEnv subclass from basic.py
    from basic import AntEnv

    print(f"Creating environment from: {args.xml_path}")
    env = AntEnv(args.xml_path)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"obs_dim={obs_dim}  act_dim={act_dim}")

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    agent = PPO(
        obs_dim        = obs_dim,
        act_dim        = act_dim,
        lr             = args.lr,
        gamma          = args.gamma,
        lam            = args.lam,
        clip_eps       = args.clip_eps,
        n_steps        = args.n_steps,
        n_minibatches  = args.n_minibatches,
        n_epochs       = args.n_epochs,
        vf_coef        = args.vf_coef,
        ent_coef       = args.ent_coef,
        device         = device,
    )

    obs, _ = env.reset()
    total_steps = 0
    update = 0

    while total_steps < args.total_steps:
        obs, ep_rewards, ep_lengths = agent.collect_rollout(env, obs)
        total_steps += args.n_steps
        update += 1

        stats = agent.update()

        if ep_rewards:
            mean_return = np.mean(ep_rewards)
            mean_len    = np.mean(ep_lengths)
        else:
            mean_return = mean_len = float("nan")

        print(
            f"[Update {update:4d}] steps={total_steps:8d} "
            f"ep_return={mean_return:8.2f}  ep_len={mean_len:6.1f}  "
            f"pg_loss={stats['pg_loss']:.4f}  vf_loss={stats['vf_loss']:.4f}  "
            f"entropy={stats['entropy']:.4f}"
        )

    # ------------------------------------------------------------------
    # Final evaluation — the fitness score for this robot morphology
    # ------------------------------------------------------------------
    eval_return = agent.evaluate_policy(env, n_episodes=args.eval_episodes)
    print(f"\n=== Evaluation over {args.eval_episodes} episodes ===")
    print(f"Mean return: {eval_return:.2f}")

    if args.save_path:
        agent.save(args.save_path)

    env.close()
    return eval_return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path",      type=str,   default="assets/base_ant_flat.xml")
    parser.add_argument("--total_steps",   type=int,   default=500_000)
    parser.add_argument("--n_steps",       type=int,   default=2048,
                        help="Steps per rollout (per PPO update)")
    parser.add_argument("--n_minibatches", type=int,   default=32)
    parser.add_argument("--n_epochs",      type=int,   default=10)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--lam",           type=float, default=0.95)
    parser.add_argument("--clip_eps",      type=float, default=0.2)
    parser.add_argument("--vf_coef",       type=float, default=0.5)
    parser.add_argument("--ent_coef",      type=float, default=0.0)
    parser.add_argument("--eval_episodes", type=int,   default=5)
    parser.add_argument("--save_path",     type=str,   default=None,
                        help="Path to save trained model weights")
    parser.add_argument("--cuda",          action="store_true")

    args = parser.parse_args()
    train(args)
