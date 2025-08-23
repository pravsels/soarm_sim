#!/usr/bin/env python3

import os
os.environ['MUJOCO_GL'] = 'glfw'

import mujoco
import mujoco.viewer
import numpy as np
import time
    
def main():
    # Path to model 
    model_path = "robot_models/so100/scene.xml"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Current directory: {os.getcwd()}")
        print("Make sure you're running from the correct directory!")
        return
    
    print("Loading SO ARM 100 model...")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"✓ Model loaded successfully!")
    print(f"  - Joints: {model.njnt}")
    print(f"  - Actuators: {model.nu}")
    print(f"  - Bodies: {model.nbody}")
    
    # Set home position
    home_pos = [0, -1.57, 1.57, 1.57, -1.57, 0]
    if model.nu >= len(home_pos):
        data.ctrl[:len(home_pos)] = home_pos
    
    print("\nLaunching viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Space: Pause/play")
    print("  - ESC: Exit")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep the simulation running
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer with simulation
            viewer.sync()
            
            # Small delay to control frame rate
            time.sleep(0.01)
    
    print("Viewer closed.")

if __name__ == "__main__":
    main()

