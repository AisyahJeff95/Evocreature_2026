import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import time
import pybullet_data
from scipy.spatial.transform import Rotation

direct = p.connect(p.GUI) 
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')
hoge=p.loadURDF("cube_collisionfilter.urdf", [0, 0, 0.5],p.getQuaternionFromEuler([0,0,-np.pi/2]))
p.loadURDF("r2d2.urdf", [2, 0, 0.5])

p.setRealTimeSimulation(0)
for i in range(10):
    p.stepSimulation()

width = 10
height = 5#64*3
fov = 60
aspect = width / height
near = 0.01
far = 100

rot= Rotation.from_quat(p.getBasePositionAndOrientation(hoge)[1])

cameraPos = p.getBasePositionAndOrientation(hoge)[0]
camTargetPos =rot.apply([0,100,1])
cameraUp = [0, 0, 1]

view_matrix=p.computeViewMatrix(cameraPos,camTargetPos,cameraUp)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Get depth values using the OpenGL renderer
images = p.getCameraImage(width,
                          height,
                          view_matrix,
                          projection_matrix,
                          shadow=False,
                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
rgb_opengl = np.reshape(images[2], (height, width, 4))/255
gray_opengl = 0.299 * rgb_opengl[:, :, 2] + 0.587 * rgb_opengl[:, :, 1] + 0.114 * rgb_opengl[:, :, 0]
# Plot both images - should show depth values of 0.45 over the cube and 0.5 over the plane
plt.imshow(rgb_opengl)
plt.show()
