from creature.creature import create
import neat
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import pickle
from tqdm import tqdm
import datetime
import shutil
import copy
import glob
import creature.visualize as visualize
import random

comment="評価関数変更_2"

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
defaultPatrsNum=1 #初期状態のパーツ数
maxPartsNum=15 #パーツ数がこれ以上ならば発生イベントは生じない
growRate=50 # 成長の刻み幅 growstep=growRate ならそのパーツの成長は終了
growInterval=20 # 成長が起こるまでのインターバル
EventNum=4 # 発生イベントの回数 個体の持つジョイント数がPybulletで指定された閾値(127)を超えるとエラーを吐く
radarAccuracy=32
radarRange=60
Total_Step=40000

Genomes=[]
def eval_fitness(genomes,config=None,mode="DIRECT"):
  if p.isConnected()==0:
    if mode=="DIRECT":
      p.connect(p.DIRECT)
    else:
      p.connect(p.GUI)
      p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
      p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
      p.resetDebugVisualizerCamera(cameraDistance=60, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,-20,0])
      # p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[60,0,0])

  os.chdir(os.path.dirname(__file__)+"/creature")
  p.loadURDF('plane.urdf',[0,0,-50])
  p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
  cube=p.loadURDF("cube.urdf",[0,0,2.5],globalScaling=5)
  p.setGravity(0, 0, -10)
  p.setRealTimeSimulation(0)

  distance_input=[[0,0,2]]
  pos_list=[i for i in range(len(genomes))]
  for _,g in genomes:
    try:
      pos_list.remove(g.pos)
    except:
      pass
  
  for _,g in genomes:
    try:
      g.pos
    except:
      g.pos=random.choice(pos_list)   
      pos_list.remove(g.pos)   

  for _,g in genomes:
    g.creature=create() 
    g.creature.create_base(sphereRadius,[radarRange*np.cos(2*np.pi*g.pos/len(genomes)),radarRange*np.sin(2*np.pi*g.pos/len(genomes)),sphereRadius],p.getQuaternionFromEuler([0,0,2*np.pi*g.pos/len(genomes)]),defaultPatrsNum)
    input_coordinates  = [(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]+distance_input
    hidden_coordinates = [[(i, 0.0, 1) for i in range(-4,5)]]
    output_coordinates = [(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
    g.prePosition=np.array(p.getBasePositionAndOrientation(g.creature.bodyId)[0])
    if __name__ == '__main__':
      g.cppn=neat.nn.FeedForwardNetwork.create(g, config)
      # g.fitness=0
    g.substrate=Substrate(input_coordinates, output_coordinates, hidden_coordinates)
    g.net=create_phenotype_network(g.cppn, g.substrate)

  growFlag=False
  growstep=0 
  EventOppotunity=0
  for step in tqdm(range(Total_Step)):
    if np.mod(step,int(Total_Step/4))==0:
        growFlag=True
    if np.mod(step,growInterval)==0 and EventOppotunity<EventNum and growFlag==True:
      growstep+=1
      if growstep==1:
        for _,g in genomes:
          g.outputs=[]
          if p.getNumJoints(g.creature.bodyId)<maxPartsNum:
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
                  g.creature.jointGrobalPosition[n][m]=[]
                  if cppn_addORnot>=1.5:
                    g.outputs.append([cppn_scale,cppn_jointType,cppn_linkPositions,cppn_orientation,cppn_linkParentInd])
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
      for _,g in genomes:
        if g.outputs!=[]:
          # 成長イベント
          g.creature.bodyId=g.creature.grow(growstep,growRate,g.outputs)

      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    if growstep==growRate:
      EventOppotunity+=1
      growFlag=False
      for _,g in genomes:
        growstep=0
        g.substrate.input_coordinates=[(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]+distance_input
        g.substrate.output_coordinates=[(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]
        g.net=create_phenotype_network(g.cppn, g.substrate)

    for _,g in genomes:
      jointlist=[i-1 for i, key in g.creature.identification.items() if key == 'joint']
      try:
        distance=[p.getClosestPoints(g.creature.bodyId,cube,radarRange)[0][8]]
      except:
        distance=[radarRange]
      p.setJointMotorControlArray(g.creature.bodyId,
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=g.net.activate([step/Total_Step]+distance if len(jointlist)==0
                                    else [step/Total_Step]+[i[1] for i in p.getJointStates(g.creature.bodyId,jointlist)]+distance),
                                  forces=[1000]*len(jointlist))

      # g.prePosition=p.getBasePositionAndOrientation(g.creature.bodyId)[0]
        # if np.mod(step,1000)==0:
        #   p.addUserDebugText("ID:"+str(id)+"__"+str(round(g.fitness,1)),
        #                       [0, 0, 0.1],
        #                       textColorRGB=[1, 0, 0] if g.fitness<0 else [0, 0, 1],
        #                       textSize=1.5,
        #                       parentObjectUniqueId=g.creature.bodyId,
        #                       parentLinkIndex=-1)

    p.stepSimulation()

  if __name__ == '__main__':
    # if np.mod(step,1000)==0:
    #   p.removeAllUserDebugItems()
    for _,g in genomes:
      # isMove=np.sqrt((p.getBasePositionAndOrientation(g.creature.bodyId)[0][0]-g.prePosition[0])**2+(p.getBasePositionAndOrientation(g.creature.bodyId)[0][1]-g.prePosition[1])**2)
      try:
        g.fitness=radarRange-p.getClosestPoints(g.creature.bodyId,cube,radarRange)[0][8]
      except:
        g.fitness=0

  if __name__ == '__main__':
    # g.creature, g.substrate, g.net = None, None, None
    g.creature, g.outputs, g.prePosition=None, None, None
    Genomes.append(copy.deepcopy(genomes))
    save_interval=200
    if np.mod(len(Genomes),save_interval)==0:
      with open(data_dir+'/genomes_'+str(len(Genomes))+'.pkl', 'wb') as output:
        pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)
      if os.path.isfile(data_dir+'/genomes_'+str(len(Genomes)-save_interval)+'.pkl')==1:
        os.remove(data_dir+'/genomes_'+str(len(Genomes)-save_interval)+'.pkl')

  p.resetSimulation()

def run(gens):
  pop = neat.population.Population(config)
  # local_dir=os.path.dirname(__file__)
  # pop=neat.Checkpointer.restore_checkpoint(local_dir+"/neat-checkpoint-1000")
  # def restore_checkpoint(filename):
  #   import gzip, random
  #   """Resumes the simulation from a previous saved point."""
  #   with gzip.open(filename) as f:
  #       _, _, population, species_set, rndstate = pickle.load(f)
  #       random.setstate(rndstate)
  #       return neat.population.Population(config, (population, species_set, 94))
  # pop=restore_checkpoint(local_dir+"/neat-checkpoint-1000")

  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))
  winner = pop.run(eval_fitness, gens)
  visualize.plot_stats(stats, ylog=False, view=True)
  # save
  os.chdir(data_dir)
  n=neat.Checkpointer()
  n.save_checkpoint(config,pop.population,pop.species,len(Genomes)-1)
  pkls=glob.glob(data_dir+"/*.pkl")
  for pkl in pkls:
    os.remove(pkl)
  with open(data_dir+'/genomes.pkl', 'wb') as output:
    pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)
  return winner

if __name__ == '__main__':
  winner = run(1000)