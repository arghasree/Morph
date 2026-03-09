"""
Standalone script to run the Ant environment in MuJoCo and save a video.
Uses MuJoCo's built-in Ant model - no XML needed!
"""

import os
# Set environment variable for headless rendering (MUST be before importing mujoco)
os.environ['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for offscreen rendering

import mujoco
import numpy as np
import imageio
import gymnasium as gym
from gymnasium.envs.mujoco.ant_v5 import AntEnv as GymAntEnv
from xml_wrapper import Robot


class AntEnv(GymAntEnv):
    """
    Subclass of gymnasium's Ant-v5 environment.
    Inherits: step(), reset(), action_space, observation_space, and reward computation.
    Adds:     offscreen video rendering utilities.
    """

    def __init__(self, xml_path, width=640, height=480, fps=30, **kwargs):
        # Resolve absolute path to the XML file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.abspath(os.path.join(script_dir, '..', xml_path))

        # Parent handles: self.model, self.data, step(), reset(), reward, obs
        # width/height set the offscreen framebuffer size in the parent — must match or exceed renderer size
        super().__init__(xml_file=model_path, width=width, height=height, **kwargs)

        self.fps = fps
        self.camera = self._camera_setup()
        # Use a separate renderer for video recording (avoid conflict with parent's render())
        self._video_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.frames = []

    def _camera_setup(self):
        # Configure camera to zoom in on the ant
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = 2.0  # Distance from lookat point (smaller = closer zoom)
        camera.lookat[:] = [0, 0, 0.5]  # Point to look at [x, y, z]
        camera.elevation = -20  # Vertical angle (negative = looking down)
        camera.azimuth = 90  # Horizontal rotation
        return camera

    def update_scene(self):
        self._video_renderer.update_scene(self.data, camera=self.camera)
        frame = self._video_renderer.render()
        self.frames.append(frame)

    def save_video_from_frames(self, output_path):
        print(f"\nSaving video to: {output_path}")
        imageio.mimsave(output_path, self.frames, fps=self.fps)
        print(f"Video saved successfully! Total frames: {len(self.frames)}")

    def close_renderer(self):
        self._video_renderer.close()
        self.close()
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.update_scene()  # Capture frame after each step
        return obs, reward, terminated, truncated, info
        


def run_ant_random_steps(args,
    xml_path='assets/base_ant_flat.xml',
    output_path="ant_random_video.mp4",
):
    """
    Run the ant environment with random actions and save a video.
    
    Args:
        xml_path: Path to the Ant XML model
        output_path: Output video file path
        num_steps: Number of simulation steps
        fps: Frames per second for the video
        width: Video frame width
        height: Video frame height
    """
    # Load the built-in Ant model from MuJoCo
    print("Loading Ant model from MuJoCo...")
    
    
    env = AntEnv(xml_path, width=args.width, height=args.height, fps=args.fps)

    print(f"Running simulation with random actions for {args.num_steps} steps...")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    total_reward = 0.0

    for step in range(args.num_steps):
        # Sample a random action from the continuous action space
        action = env.action_space.sample()

        # step() returns gymnasium (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    print(f"Total reward over {args.num_steps} steps: {total_reward:.2f}")
    env.save_video_from_frames(output_path)
    env.close_renderer()
    
    
def change_body(args, initial_xml_path='assets/base_ant_flat.xml'):
    # ---------------------------------------------------------
    # Run the simulation
    run_ant_random_steps(
        args,
        xml_path=initial_xml_path,
        output_path="initial.mp4",
    )
    
    OUT_PATH = 'assets/tmp/ant_modified.xml'
    
    # --- 1. Load ---
    robot = Robot(initial_xml_path, 'Ant-v5')

    print('=== Initial bodies ===')
    for b in robot.bodies:
        print(f'  {b}  parent={b.parent}  pos={b.pos}  bone_start={b.bone_start}  bone_end={b.bone_end}')

    print(f'\nInitial GNN edges:\n{robot.get_gnn_edges()}\n')

    # --- 2. Add a limb to body '1' (the single existing leg) ---
    leg = robot.bodies[1]          # body named '1'
    print(f'Adding child to: {leg}')
    robot.add_child_to_body(leg, angle=np.pi / 2)
    robot.add_child_to_body(leg, angle=np.pi / 2)
    
    leg2 = robot.bodies[-1]        # the new leg we just added
    print(f'Adding child to: {leg2}')
    robot.add_child_to_body(leg2, angle=np.pi / 2)

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
    # Run the simulation
    run_ant_random_steps(
        args,
        xml_path=OUT_PATH,
        output_path='changed.mp4',
    )
    
    
    
    


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
    
    
    args = parser.parse_args()
    change_body(args, initial_xml_path='assets/base_ant_flat.xml')
       