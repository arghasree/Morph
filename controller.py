import os
from basic import AntEnv
from ppo import PPO, train
import mujoco
import numpy as np
import imageio
import gymnasium as gym
from xml_wrapper import Robot


def run(args, xml_path, trained_path, initial=False):
    env = AntEnv(xml_path)
    # if initial and args.PPO_save_path and os.path.exists(args.PPO_save_path):
    #     obs_dim = env.observation_space.shape[0]
    #     act_dim = env.action_space.shape[0]
    #     agent = PPO(obs_dim=obs_dim, act_dim=act_dim)
    #     agent.load(args.PPO_save_path)
    # else:
    agent,_ = train(args, env)
    
    # Save a video of the trained agent
    obs, info = env.reset()
    done = False
    env.reset_frames()  # Clear any frames from the random steps
    eval_return = agent.evaluate_policy(env, n_episodes=1)
    env.save_video_from_frames(trained_path)
    env.close_renderer()
    
    
def change_body(args, initial_xml_path='assets/base_ant_flat.xml'):
    # ---------------------------------------------------------
    # Run the simulation
    run(args, initial_xml_path, initial=True, trained_path='initial.mp4')
    OUT_PATH = args.directory+'assets/tmp/ant_modified.xml'
    
    # --- 1. Load ---
    robot = Robot(initial_xml_path, 'Ant-v5')

    print('=== Initial bodies ===')
    for b in robot.bodies:
        print(f'  {b}  parent={b.parent}  pos={b.pos}  bone_start={b.bone_start}  bone_end={b.bone_end}')

    print(f'\nInitial GNN edges:\n{robot.get_gnn_edges()}\n')

    # --- 2. Add a limb to body '1' (the single existing leg) ---
    leg = robot.bodies[1]          # body named '1'
    print(f'Adding child to: {leg}')
    robot.add_child_to_body(leg, angle=np.pi/2, tilt=-np.pi/4)
    robot.add_child_to_body(leg, angle=np.pi/2, tilt=-np.pi/4)
    
    leg2 = robot.bodies[-1]        # the new leg we just added
    print(f'Adding child to: {leg2}')
    robot.add_child_to_body(leg2, angle=np.pi/2, tilt=0)
    robot.add_child_to_body(leg2, angle=np.pi/2, tilt=-np.pi/4)

    print('\n=== Bodies after add_child_to_body ===')
    for b in robot.bodies:
        print(f'  {b}  parent={b.parent}  pos={b.pos}')

    print(f'\nNew GNN edges:\n{robot.get_gnn_edges()}\n')

    # --- 3. Write modified XML ---
    if not os.path.exists('assets/tmp'):
        os.makedirs('assets/tmp')
    robot.write_xml(OUT_PATH)
    print(f'Written to {OUT_PATH}')

    # Debug: print geom fromto for each body
    for b in robot.bodies:
        for g in b.geoms:
            print(f'  {b} geom start={g.start}  end={g.end}  fromto_in_xml={g.node.attrib.get("fromto", "N/A")}')
    
    # ---------------------------------------------------------
    run(args, OUT_PATH, initial=False, trained_path='modified.mp4')
    
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Ant environment with random actions and save video")
    parser.add_argument("--output", type=str, default="/home/arghasre/scratch/CMPUT 605/BodyGen/arghasre/ant_random_video.mp4",
                        help="Output video file path")
    parser.add_argument("--num_steps", type=int, default=300,
                        help="Number of simulation steps (default: 300)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--width", type=int, default=640,
                        help="Video width")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height")
    # PPO hyperparameters
    parser.add_argument("--PPO_lr", type=float, default=3e-4,
                        help="Learning rate for PPO optimizer")
    parser.add_argument("--PPO_gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--PPO_lam", type=float, default=0.95,
                        help="GAE lambda for advantage estimation")
    parser.add_argument("--PPO_clip_eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--PPO_n_steps", type=int, default=100,
                        help="Number of steps to collect per rollout")
    parser.add_argument("--PPO_n_minibatches", type=int, default=32,
                        help="Number of minibatches per PPO epoch")
    parser.add_argument("--PPO_n_epochs", type=int, default=10,
                        help="Number of PPO epochs per update")
    parser.add_argument("--PPO_vf_coef", type=float, default=0.5,
                        help="Value function loss coefficient")
    parser.add_argument("--PPO_ent_coef", type=float, default=0.0,
                        help="Entropy bonus coefficient")
    parser.add_argument("--PPO_total_steps", type=int, default=100,
                        help="Total environment steps to train for")
    parser.add_argument("--PPO_eval_episodes", type=int, default=5,
                        help="Number of episodes for policy evaluation")
    parser.add_argument("--PPO_save_path", type=str, default=None,
                        help="Path to save/load the trained PPO model")
    parser.add_argument("--directory", type=str, default='/home/arghasre/scratch/arghasre/',)

    args = parser.parse_args()
    change_body(args, initial_xml_path=args.directory+'assets/base_ant_flat.xml')