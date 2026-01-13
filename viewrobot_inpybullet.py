import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
p.connect(p.GUI)

# Set search path for PyBullet data (for plane.urdf, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Optional: set gravity
p.setGravity(0, 0, -9.81)

# Load a ground plane
plane_id = p.loadURDF("plane.urdf")

# Load your URDF object
# Replace this with the path to YOUR urdf file
robot_id = p.loadURDF(
    "r2d2.urdf",        # example URDF
    basePosition=[0, 0, 0.1],
    useFixedBase=False
)

# Keep the simulation running
while True:
    p.stepSimulation()
    time.sleep(1 / 240)
