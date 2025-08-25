# mujoco_viewer.py 

import os
os.environ['MUJOCO_GL'] = 'glfw'

import zmq
import json 
import mujoco
import mujoco.viewer
import numpy as np
import time

# port for publishing sim arm 
PUB_ADDR = "tcp://*:6001"
# port for subscribing to real arm 
SUB_ADDR = "tcp://localhost:6000"
    
def main():
    # Path to model 
    model_path = "robot_models/so101/scene.xml"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print(f"Current directory: {os.getcwd()}")
        print("Make sure you're running from the correct directory!")
        return
    
    print("Loading robot model ...")
    
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Model loaded successfully!")
    print(f"  - Joints: {model.njnt}")
    print(f"  - Actuators: {model.nu}")
    print(f"  - Bodies: {model.nbody}")

    # ZeroMQ for publisher and subscriber 
    ctx = zmq.Context()

    pub = ctx.socket(zmq.PUB)
    pub.bind(PUB_ADDR)
    print(f"Publishing sim arm's joint states on {PUB_ADDR}, topic = so101.state_sim")

    sub = ctx.socket(zmq.SUB)
    sub.connect(SUB_ADDR)
    sub.setsockopt(zmq.SUBSCRIBE, b"so101.state_real")
    print(f"Subscribing for real arm's joint states on {SUB_ADDR}, topic = so101.state_real")

    latest_real_qpos = None 
    
    # Set home position
    home_pos = [0, 0, 0, 0, 0, 0]
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

        try: 
            # Keep the simulation running
            while viewer.is_running():                
                # check if a message is waiting 
                if sub.poll(timeout=0): 
                    topic, payload = sub.recv_multipart(flags=zmq.NOBLOCK)
                    msg = json.loads(payload.decode())
                    latest_real_qpos = np.array(msg['qpos'], dtype=np.float32)

                # if we have a real state, apply it to the sim arm 
                if latest_real_qpos is not None: 
                    n = min(model.nq, len(latest_real_qpos))
                    data.qpos[:n] = latest_real_qpos[:n]
                    mujoco.mj_forward(model, data)

                # Step the simulation
                mujoco.mj_step(model, data)

                # Publish joint angles 
                qpos = data.qpos[:model.nq].astype(float).tolist()
                msg = {"t": time.time(), "qpos": qpos}
                pub.send_multipart([b"so101.state_sim", json.dumps(msg).encode()])
                
                # Sync viewer with simulation
                viewer.sync()
                time.sleep(0.02)  # ~50Hz

        except KeyboardInterrupt:
            pass 
        finally: 
            pub.close(0)
            sub.close(0)
            ctx.term()
    
    print("Viewer closed.")

if __name__ == "__main__":
    main()

