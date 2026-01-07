from creature.c_test import create
import neat
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import pybullet as p
import pybullet_data
import numpy as np
import os
import pickle
from tqdm import tqdm
import datetime
import shutil
import copy
import glob
import creature.visualize as visualize
comment=None

config_path = os.path.dirname(__file__)+'/config_cppn'
def create_data_dir(comment=None):
  data_dir = os.path.dirname(__file__)
  if comment==None:
    now = datetime.datetime.now()
    data_dir=data_dir+"/exp_data/"+os.path.basename(__file__)+"/"+now.strftime("%Y%m%d_%H%M%S")
  else:
    data_dir=data_dir+"/exp_data/"+os.path.basename(__file__)+"/"+comment  
  os.makedirs(data_dir)
  os.chdir(data_dir)
  shutil.copytree(os.path.dirname(__file__)+"/creature",data_dir+"/creature")
  shutil.copy(config_path,data_dir+"/creature")
  shutil.copy(os.path.abspath(__file__),data_dir+"/evaluation.py")
  shutil.move(data_dir+"/creature/replay.py",data_dir)

  return data_dir

config_path = os.path.dirname(__file__)+'/config_cppn'
if __name__ == '__main__':
  config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                              config_path)
  # data_dir=create_data_dir(comment)

sphereRadius=0.3 # 仮想生物の長方形の短軸長
defaultPatrsNum=2 #初期状態のパーツ数
radarRange=50
Total_Step=20000

mode="DIRECT"
if mode=="DIRECT":
  p.connect(p.DIRECT)
else:
  p.connect(p.GUI)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadSDF("stadium.sdf",globalScaling=10)
# os.chdir(os.path.dirname(__file__)+"/creature")
# p.loadURDF('plane.urdf',[0,0,-50])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=120, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
# p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[60,0,0])
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

Genomes=[]
def eval_fitness(genomes,config=None):
  pos=0
  for _,g in genomes:
    for step in range(Total_Step):
      if step==0:
        g.creature=create(sphereRadius,[(radarRange-10)*np.cos(2*np.pi*pos/len(genomes)),(radarRange-10)*np.sin(2*np.pi*pos/len(genomes)),sphereRadius+0.01],p.getQuaternionFromEuler([0,0,2*np.pi*pos/len(genomes)]),defaultPatrsNum)
        g.input_coordinates  = [(0,-1,0),(0,1,0)]#[(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]#+radar_matrix
        g.hidden_coordinates = [[(i, 0.0, 1) for i in range(-4,5)]]
        g.output_coordinates = [(0,0,1)]#[(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
        g.prePosition=np.array(p.getBasePositionAndOrientation(g.creature.bodyId)[0])
        g.fitness=0
        g.cppn=neat.nn.FeedForwardNetwork.create(g, config)
        g.substrate=Substrate(g.input_coordinates, g.output_coordinates, g.hidden_coordinates)
        g.net=create_phenotype_network(g.cppn, g.substrate)
      
      jointlist=[i-1 for i, key in g.creature.identification.items() if key == 'joint']
      targetPositions=g.net.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[round(i[1],1) for i in p.getJointStates(g.creature.bodyId,jointlist)])
      # targetVelocities=g.net.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[i[1] for i in p.getJointStates(g.creature.bodyId,jointlist)])
      # targetPositions=g.net.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[0.5]*len(jointlist))
      p.setJointMotorControlArray(g.creature.bodyId,
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=targetPositions,
                                  forces=[1000]*len(jointlist))
      p.stepSimulation()
    g.fitness=np.linalg.norm(np.array(p.getBasePositionAndOrientation(g.creature.bodyId)[0])-g.prePosition)
    p.removeBody(g.creature.bodyId)
    pos+=1

  Genomes.append(copy.deepcopy(genomes))

def run(gens):
  pop = neat.population.Population(config)
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))
  pop.run(eval_fitness, gens)
  visualize.plot_stats(stats, ylog=False, view=True)

  # # save
  # os.chdir(data_dir)
  # n=neat.Checkpointer()
  # n.save_checkpoint(config,pop.population,pop.species,gens)
  # pkls=glob.glob(data_dir+"/*.pkl")
  # for pkl in pkls:
  #   os.remove(pkl)
  # with open(data_dir+'/genomes.pkl', 'wb') as output:
  #   pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  winner = run(50)