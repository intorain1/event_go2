# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This script demonstrates how to add and simulate on-board sensors for a robot.
We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):
* USD-Camera: This is a camera sensor that is attached to the robot's base.
* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.
"""
"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="tutorial of for ")
parser.add_argument("--num_envs", type=int, default=2, help="env_nums")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from pynput import keyboard 
import threading
from PIL import Image
import os

root_vel = torch.zeros((1, 6), dtype=torch.float32)
key_pressed = False
lock = threading.Lock()

def design_scene():
    cfg_board = sim_utils.UsdFileCfg(usd_path='/media/mspx/Elements1/calibration.usd', scale=(0.001,0.001,0.001))
    cfg_board.func("/World/Objects/Cboard", cfg_board, translation=(2.0, 0.0, 0.0), orientation=(0.0, 0.707, 0.0, 0.707))

def on_press(key):
    global root_vel, key_pressed
    try:
        with lock:
            if key.char == 'w':
                root_vel[:, 0] = 2.0 
                key_pressed = True
            elif key.char == 's':
                root_vel[:, 0] = -2.0 
                key_pressed = True
            elif key.char == 'a':
                root_vel[:, 1] = -2.0  
                key_pressed = True
            elif key.char == 'd':
                root_vel[:, 1] = 2.0 
                key_pressed = True
    except AttributeError:
        pass

def on_release(key):
    global root_vel, key_pressed
    with lock:
        key_pressed = False
        root_vel[:] = 0.0 
    if key == keyboard.Key.esc:
        return False 

@configclass
class SensorsSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    board = AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/Cboard',
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(2.0, 0.0, 0.0),
            rot=(0.0, 0.707, 0.0, 0.707)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path='/media/mspx/Elements1/calibration.usd',
            scale=(0.001, 0.001, 0.001)
        )
    )
    
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while simulation_app.is_running():
        if count % 50000 == 0:
            count = 0
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO]: reset robot...")

        with lock:
            current_vel = root_vel.clone().to(scene["robot"]._device)
        
        if key_pressed:
            scene["robot"].write_root_velocity_to_sim(current_vel)
        else:
            scene["robot"].write_root_velocity_to_sim(torch.zeros_like(current_vel))
        
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        scene.write_data_to_sim()
        
        img = scene["camera"].data.output["rgb"][0]
        # print(f"image shape: {img.shape}")
        # Create a directory to save images if it doesn't exist
        output_dir = "/home/mspx/IsaacLab/test_images"
        os.makedirs(output_dir, exist_ok=True)

        # Convert the image to a PIL Image and save it
        img_pil = Image.fromarray(img.cpu().numpy())
        img_pil.save(os.path.join(output_dir, f"frame_{count}.png"))

        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

        print(f"velocity: {current_vel[0,:3].cpu().numpy()} | keyboard state: {'activate' if key_pressed else 'free'}")

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.001, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    
    # design_scene()

    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: set done...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()