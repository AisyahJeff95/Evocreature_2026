from creature.creature import create
import neat
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import pybullet as p
import pybullet_data
import time
import numpy as np
from collections import defaultdict
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
  
# Config for CPPN.
if __name__ == '__main__':
  config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                              config_path)
  data_dir=create_data_dir(comment)

sphereRadius=0.3 # 仮想生物の長方形の短軸長
defaultPatrsNum=2 #初期状態のパーツ数
maxPartsNum=5 #パーツ数がこれ以上のときは発生イベントは生じない
growRate=50 # 成長の刻み幅 growstep=growRate ならそのパーツの成長は終了
growInterval=400 # 成長が起こるまでのインターバル
EventNum=0 # 発生イベントの回数 個体の持つジョイント数がPybulletで指定された閾値(127)を超えるとエラーを吐く
radarAccuracy=32
radarRange=60
Total_Step=40000

mode="DIRECT"
if mode=="DIRECT":
  p.connect(p.DIRECT)
else:
  p.connect(p.GUI)
  p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
os.chdir(os.path.dirname(__file__)+"/creature")
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=60, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
# p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[60,0,0])
p.loadURDF('plane.urdf',[0,0,-50])
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)


Genomes=[]
def eval_fitness(genomes,config=None,mode="DIRECT",sleep=0):
  growFlag=False
  pos=0
  for _,g in genomes:
    for step in range(Total_Step):
      if step==0:
        if not "g.creature" in locals():
          g.creature=create() 
        g.creature.create_base(sphereRadius,[(radarRange-10)*np.cos(2*np.pi*pos/len(genomes)),(radarRange-10)*np.sin(2*np.pi*pos/len(genomes)),sphereRadius],p.getQuaternionFromEuler([0,0,2*np.pi*pos/len(genomes)]),defaultPatrsNum)
        input_coordinates  = [(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]#+radar_matrix
        hidden_coordinates = [[(i, 0.0, 1) for i in range(-4,5)]]
        output_coordinates = [(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
        g.prePosition=np.array([(radarRange-10)*np.cos(2*np.pi*pos/len(genomes)),(radarRange-10)*np.sin(2*np.pi*pos/len(genomes)),sphereRadius])
        g.growstep=0 
        g.EventOppotunity=0

        if __name__ == '__main__':
          g.fitness=0
          g.cppn=neat.nn.FeedForwardNetwork.create(g, config)
        g.substrate=Substrate(input_coordinates, output_coordinates, hidden_coordinates)
        g.net=create_phenotype_network(g.cppn, g.substrate)

      if np.mod(step,int(Total_Step/4))==0:
          growFlag=True
      if np.mod(step,growInterval)==0 and g.EventOppotunity<EventNum and growFlag==True:
        g.growstep+=1
        if g.growstep==1:
          g.outputs=[]
          if p.getNumJoints(g.creature.bodyId)<maxPartsNum: #pybulletのjoint数上限が127
            input=list(g.creature.input_coordinate.keys())
            for n in input:
              for m in [0,1,2]:
                if len(g.creature.jointGrobalPosition[n][m])!=0:
                  cppn_output=g.cppn.activate(7*[0]+[step/Total_Step]+list(g.creature.input_coordinate[n])+list(g.creature.jointGrobalPosition[n][m]))
                  cppn_addORnot=cppn_output[1]
                  cppn_scale=4 if cppn_output[2]<4 else cppn_output[2] if 4<cppn_output[2] and cppn_output[2]<8 else 8
                  cppn_jointType=p.JOINT_REVOLUTE if cppn_output[3]>=0 else p.JOINT_FIXED
                  cppn_orientation=p.getQuaternionFromEuler(cppn_output[4:])
                  cppn_linkParentInd=n
                  cppn_linkPositions=g.creature.jointLinkPosition[n][m]
                  if cppn_addORnot>=1:
                    g.outputs.append([cppn_scale,cppn_jointType,cppn_linkPositions,cppn_orientation,cppn_linkParentInd])
                    g.creature.jointGrobalPosition[n][m]=[]

        if g.outputs!=[]:
          # 成長イベント
          p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
          g.creature.bodyId=g.creature.grow(g.growstep,growRate,g.outputs)
          p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        if g.growstep==growRate:
          g.EventOppotunity+=1
          growFlag=False
          g.growstep=0

          g.substrate.input_coordinates=[(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]#+radar_matrix
          g.substrate.output_coordinates=[(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
          g.net=create_phenotype_network(g.cppn, g.substrate)


      jointlist=[i-1 for i, key in g.creature.identification.items() if key == 'joint']
    
      targetPositions=g.net.activate([step/Total_Step] if len(jointlist)==0 else [step/Total_Step]+[i[1] for i in p.getJointStates(g.creature.bodyId,jointlist)])
      p.setJointMotorControlArray(g.creature.bodyId,
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=targetPositions,
                                  forces=[1000]*len(jointlist))

      # if __name__ == '__main__':
      #   isMove=np.sqrt((p.getBasePositionAndOrientation(g.creature.bodyId)[0][0]-g.prePosition[0])**2+(p.getBasePositionAndOrientation(g.creature.bodyId)[0][1]-g.prePosition[1])**2)
      #   if isMove>5*10**-5:
      #     g.fitness+=100/Total_Step
      #   else:
      #     g.fitness+=-200/Total_Step

      #   g.prePosition=p.getBasePositionAndOrientation(g.creature.bodyId)[0]

      p.stepSimulation()

      # p.resetSimulation()

    g.fitness=np.linalg.norm(p.getBasePositionAndOrientation(g.creature.bodyId)[0]-g.prePosition)
    p.removeBody(g.creature.bodyId)
    # pos+=1

  if __name__ == '__main__':
    # g.creature, g.substrate, g.net = None, None, None
    # g.creature, g.outputs, g.prePosition, g.radar=None, None, None, None
    Genomes.append(copy.deepcopy(genomes))
    save_interval=200
    if np.mod(len(Genomes),save_interval)==0:
      with open(data_dir+'/genomes_'+str(len(Genomes))+'.pkl', 'wb') as output:
        pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)
      if os.path.isfile(data_dir+'/genomes_'+str(len(Genomes)-save_interval)+'.pkl')==1:
        os.remove(data_dir+'/genomes_'+str(len(Genomes)-save_interval)+'.pkl')

def run(gens):
  pop = neat.population.Population(config)
  # pop=neat.Checkpointer.restore_checkpoint(local_dir+"/neat-checkpoint-1")
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))
  winner = pop.run(eval_fitness, gens)
  visualize.plot_stats(stats, ylog=False, view=True)

  # save
  os.chdir(data_dir)
  n=neat.Checkpointer()
  n.save_checkpoint(config,pop.population,pop.species,gens)
  pkls=glob.glob(data_dir+"/*.pkl")
  for pkl in pkls:
    os.remove(pkl)
  with open(data_dir+'/genomes.pkl', 'wb') as output:
    pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)
  return winner

if __name__ == '__main__':
  winner = run(50)