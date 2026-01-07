from creature.creature import create
import neat
from pureples.shared.visualize import draw_net_3d
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import pybullet as p
import time
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from collections import defaultdict
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

sphereRadius=0.3 # 仮想生物の長方形の短軸長
defaultPatrsNum=1 #初期状態のパーツ数
growRate=20 # 成長の刻み幅 growstep=growRate ならそのパーツの成長は終了
growWaiting=250 # 成長が起こるまでのインターバル
EventNum=4 # 発生イベントの回数 個体の持つジョイント数がPybulletで指定された閾値(127)を超えるとエラーを吐く

# Config for CPPN.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_cppn_evo')
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            config_path)

fitnesses=[]
Genomes={}
def eval_fitness(genomes,config,mode="GUI",sleep=0):
  if mode=="DIRECT":
    p.connect(p.DIRECT)
  else:
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,0)
  os.chdir(local_dir+"/creature")
  p.loadURDF('plane.urdf',[0,0,0])
  # p.createCollisionShape(p.GEOM_PLANE)
  # p.createMultiBody(0, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.resetDebugVisualizerCamera(cameraDistance=70, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[0,-20,0])
  # p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=0, cameraPitch=-50, cameraTargetPosition=[60,0,0])

  p.setGravity(0, 0, -10)
  p.setRealTimeSimulation(0)

  creatures={} 
  creatures["info"]=[] #各仮想生物の情報を保存
  creatures["id"]=[] #各仮想生物のIDを保存
  cppns=[]
  nets=[]
  ges=[]
  ind=0
  #eye_matrix=[[i,j,-0.2] for i in range(64) for j in range(64)]
  
  for _,g in genomes:
    creatures["info"].append(create())
    creatures["id"].append(creatures["info"][ind].create_base(sphereRadius,[60*np.cos(2*np.pi*ind/len(genomes)),60*np.sin(2*np.pi*ind/len(genomes)),sphereRadius],defaultPatrsNum))
    input_coordinates  = [(creatures["info"][ind].input_coordinate[i]) for i in creatures["info"][ind].input_coordinate]
    hidden_coordinates = [[(0.0, 0.0, 0.2)]]
    output_coordinates = [(creatures["info"][ind].output_coordinate[i]) for i in creatures["info"][ind].output_coordinate]
    sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

    cppns.append(neat.nn.FeedForwardNetwork.create(g, config))
    g.substrate=copy.deepcopy(sub)
    ges.append(g)
    nets.append(create_phenotype_network(cppns[ind], g.substrate))
    ind+=1

  Total_Step=growWaiting*growRate*EventNum*2
  growstep=0 
  EventOppotunity=0
  for step in tqdm(range(Total_Step)):
    # print(step)
    if np.mod(step,growWaiting)==0 and EventOppotunity<EventNum:
      growstep+=1
      if growstep==1:
        outputs=defaultdict(list)
        for ind, c in enumerate(creatures["info"]):
          if p.getNumJoints(creatures["id"][ind])<24: #pybulletのjoint数上限が127
            input=list(c.input_coordinate.keys())
            for n in input:
              for m in [0,1,2]:
                if len(c.jointGrobalPosition[n][m])!=0:
                  cppn_output=cppns[ind].activate(7*[0]+[step/Total_Step]+list(c.input_coordinate[n])+list(c.jointGrobalPosition[n][m]))
                  cppn_addORnot=cppn_output[1]
                  cppn_scale=4 if cppn_output[2]<4 else cppn_output[2] if 4<cppn_output[2] and cppn_output[2]<8 else 8
                  cppn_jointType=p.JOINT_REVOLUTE if cppn_output[3]>=0 else p.JOINT_FIXED
                  cppn_orientation=p.getQuaternionFromEuler(cppn_output[4:])
                  cppn_linkParentInd=n
                  cppn_linkPositions=c.jointLinkPosition[n][m]
                  c.jointGrobalPosition[n][m]=[]
                  if cppn_addORnot>=0:
                    outputs[ind].append([cppn_scale,cppn_jointType,cppn_linkPositions,cppn_orientation,cppn_linkParentInd])
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
      for ind, c in enumerate(creatures["info"]):
        # c.eye(creatures["id"][ind])
        if outputs[ind]!=[]:
          # 成長イベント
          creatures["id"][ind]=c.grow(creatures["id"][ind],growstep,growRate,outputs[ind])
          # アタリ判定
          # partsInd=p.getNumJoints(creatures["id"][ind])
          # for i in range(-1,partsInd):
          #   for j in range(i+1, partsInd):
          #     p.setCollisionFilterPair(creatures["id"][ind], creatures["id"][ind], i,j, enableCollision=1)
      p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    if growstep==growRate:
      EventOppotunity+=1
      for ind, c in enumerate(creatures["info"]):
        growstep=0
        ges[ind].substrate.input_coordinates=[(c.input_coordinate[i]) for i in c.input_coordinate]
        ges[ind].substrate.output_coordinates=[(c.output_coordinate[i]) for i in c.output_coordinate]
        nets[ind]=create_phenotype_network(cppns[ind], ges[ind].substrate)

    for ind, c in enumerate(creatures["info"]):
      jointlist=[i-1 for i, key in c.identification.items() if key == 'joint']
                                          
      p.setJointMotorControlArray(creatures["id"][ind],
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=nets[ind].activate([step/Total_Step] if len(jointlist)==0
                                    else [step/Total_Step]+[i[1] for i in p.getJointStates(creatures["id"][ind],jointlist)]),
                                  forces=[200]*len(jointlist))

    p.stepSimulation()
    time.sleep(sleep)

  for ind, g in enumerate(ges):
    g.fitness=np.sqrt((np.array(creatures["info"][ind].originPosition)[0]-np.array(p.getBasePositionAndOrientation(creatures["id"][ind])[0][0]))**2
                        +(np.array(creatures["info"][ind].originPosition)[1]-np.array(p.getBasePositionAndOrientation(creatures["id"][ind])[0][1]))**2)
  Genomes[len(Genomes)]=genomes

  p.disconnect()

def run(gens):
  pop = neat.population.Population(config)
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))

  winner = pop.run(eval_fitness, gens)
  print("hyperneat done")
  return winner, stats

if __name__ == '__main__':
  winner = run(1000)[0]
  print('\nBest genome:\n{!s}'.format(winner))

  # Verify network output against training data.
  print('\nOutput:')
  cppn = neat.nn.FeedForwardNetwork.create(winner, config)
  winner_net = create_phenotype_network(cppn, winner.substrate)
  
  filename = os.path.join(local_dir, 'winner.pkl')
  # Save CPPN if wished reused and draw it to file along with the winner.
  with open(filename, 'wb') as output:
      pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)
  filename = os.path.join(local_dir, 'winner_cppn')
  draw_net(cppn, filename)
  # draw_net_3d(winner_net, winner.substrate ,filename="hyperneat_xor_winner")

  filename = os.path.join(local_dir, 'genomes.pkl')
  with open(filename, 'wb') as output:
      pickle.dump(Genomes, output, pickle.HIGHEST_PROTOCOL)