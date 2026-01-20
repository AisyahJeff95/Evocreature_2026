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


# time.sleep(1./240.)
