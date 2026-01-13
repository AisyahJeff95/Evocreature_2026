import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI) # or p.DIRECT for no GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0]) # Load your robot model

# usually this remain constant:
width, height = 480, 360
fov = 60
aspect = width / height
nearVal, farVal = 0.01, 10
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearVal, farVal)

def get_camera_image(robot_id, target_link_index, distance=1.0, yaw=50, pitch=-35):
    # Get the position and orientation of the target link
    link_state = p.getLinkState(robot_id, target_link_index)
    target_pos = link_state[0]
    target_orn = link_state[1]
    
    # Compute camera position using spherical coordinates
    cam_eye_pos = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target_pos,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0,
        upAxisIndex=2
    )
    
    # Get the view matrix
    viewMatrix = cam_eye_pos
    
    # Capture the image
    img_arr = p.getCameraImage(width, height, viewMatrix, projectionMatrix)
    
    # Extract the RGB data
    rgb_array = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    
    return rgb_array
# Example usage
while True:
    rgb_image = get_camera_image(robot_id, target_link_index=6) # Assuming link index 6 is the end-effector
    # Process rgb_image as needed
    time.sleep(1./240.)
