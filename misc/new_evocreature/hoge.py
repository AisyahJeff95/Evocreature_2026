import time
import math
import numpy as np
import pybullet as p


# Constants
TIME_STEP = 1.0 / 600.

KP = 0.015
KD = 1


# Connect
p.connect( p.GUI )

# Debug config
p.configureDebugVisualizer( p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1 )
p.configureDebugVisualizer( p.COV_ENABLE_MOUSE_PICKING, 1 )

# World settings
p.setGravity( 0, 0, -9.8 )
p.setPhysicsEngineParameter( fixedTimeStep=TIME_STEP, numSolverIterations=30, numSubSteps=1 )

# Camera position
p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[4.0, 0.0, 3.5] )


def sleep( time_step ):
  real_time = time.time()
  cur_time = real_time
  real_time += time_step

  while cur_time < real_time:
    time.sleep( 0.001 )
    cur_time = time.time()


def eulerToQuaternion( yaw, pitch, roll ):
  qx = np.sin(yaw/2) * np.sin(pitch/2) * np.cos(roll/2) + np.cos(yaw/2) *  np.cos(pitch/2) * np.sin(roll/2)
  qy = np.sin(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) + np.cos(yaw/2) *  np.sin(pitch/2) * np.sin(roll/2)
  qz = np.cos(yaw/2) * np.sin(pitch/2) * np.cos(roll/2) - np.sin(yaw/2) *  np.cos(pitch/2) * np.sin(roll/2)
  qw = np.cos(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) - np.sin(yaw/2) *  np.sin(pitch/2) * np.sin(roll/2)

  return (qx, qy, qz, qw)


# Load model
model = p.loadURDF( "./humanoid.urdf", [0, 0, 4.0], [0.707, 0, 0, 0.707], globalScaling=1.0, flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION )

num_joints = p.getNumJoints( model )
pos, orn = p.getBasePositionAndOrientation( model )

# Load floor
p.createCollisionShape( p.GEOM_PLANE )
p.createMultiBody( 0, 0 )


# Parse joints
joint_names_to_id = {}
  
for i in range( num_joints ):
  info = p.getJointInfo( model, i )
  link_name = info[12].decode( "ascii" )

  joint_id = info[0]
  joint_type = info[2]

  if joint_type == p.JOINT_SPHERICAL:
    joint_names_to_id[info[1].decode('UTF-8')] = joint_id

  elif joint_type in [p.JOINT_PRISMATIC, p.JOINT_REVOLUTE]:
    joint_names_to_id[info[1].decode('UTF-8')] = joint_id


p.resetJointStateMultiDof( model, joint_names_to_id['right_shoulder'], eulerToQuaternion(0, -math.pi, 0) )
#p.resetJointState( model, joint_names_to_id['right_elbow'], math.pi/2 )

# Step loop
while True:

  # Press q to quit
  keys = p.getKeyboardEvents()
  if 113 in keys:
    break

  p.resetBasePositionAndOrientation( model, pos, orn )

  p.stepSimulation()
  sleep( TIME_STEP )

