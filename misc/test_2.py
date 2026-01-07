from creature.c_test2 import create
import neat
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import pybullet as p
import pybullet_data
import numpy as np
import os
from tqdm import tqdm
import creature.visualize as visualize
import copy

config_path = os.path.dirname(__file__)+'/config_cppn'
if __name__ == '__main__':
  config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                              config_path)

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

def eval_fitness(genomes,config):
  pos=0
  ges=[]
  for step in range(Total_Step):
    for _,g in genomes:
      if step==0:
        input_coordinates  = [(0,-1,0),(0,1,0)]#[(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]#+radar_matrix
        hidden_coordinates = [[(i, 0.0, 1) for i in range(-4,5)]]
        output_coordinates = [(0,0,1)]#[(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
        g.creature=create(g,config,Substrate(input_coordinates, output_coordinates, hidden_coordinates))
        g.creature.create_base(sphereRadius,[(radarRange-10)*np.cos(2*np.pi*pos/len(genomes)),(radarRange-10)*np.sin(2*np.pi*pos/len(genomes)),sphereRadius+0.01],p.getQuaternionFromEuler([0,0,2*np.pi*pos/len(genomes)]),defaultPatrsNum)
        g.prePosition=np.array(p.getBasePositionAndOrientation(g.creature.bodyId)[0])
        g.fitness=0
        # g.cppn=neat.nn.FeedForwardNetwork.create(g, config)
        # g.substrate=Substrate(g.input_coordinates, g.output_coordinates, g.hidden_coordinates)
        # g.net=create_phenotype_network(g.creature.genome, g.substrate)
        pos+=1
      jointlist=[i-1 for i, key in g.creature.identification.items() if key == 'joint']
      targetPositions=g.creature.phenotype.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[i[1] for i in p.getJointStates(g.creature.bodyId,jointlist)])
      # targetPositions=g.net.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[0.5]*len(jointlist))
      p.setJointMotorControlArray(g.creature.bodyId,
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=targetPositions,
                                  forces=[1000]*len(jointlist))
    p.stepSimulation()
  g.fitness=np.linalg.norm(np.array(p.getBasePositionAndOrientation(g.creature.bodyId)[0])-g.prePosition)
  for g in ges:
    p.removeBody(g.creature.bodyId)
  
  for (_,g),gp in zip(genomes,ges):
    g.fitness=gp.fitness

def run(gens):
  pop = neat.population.Population(config)
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))
  pop.run(eval_fitness, gens)
  visualize.plot_stats(stats, ylog=False, view=True)

if __name__ == '__main__':
  winner = run(50)