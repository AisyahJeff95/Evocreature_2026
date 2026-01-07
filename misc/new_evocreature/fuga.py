import pybullet as p
import time
import numpy as np
import neat
import neat.nn
from pureples.shared.visualize import draw_net_3d
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network
import copy
from scipy.spatial.transform import Rotation
from collections import defaultdict
import os
import pickle
from tqdm import tqdm

class creature:
  def __init__(self):
    self.HalfExtents={} #全ブロックの半径を格納
    self.input_coordinate={}
    self.output_coordinate={}
    self.partsType={}
    self.jointsType={}
    self.identification={}
    self.jointGrobalPosition=defaultdict(list)
    self.jointLinkPosition={}
    self.linkmasses=[]
    self.matureLinkParentIndices=[]
    self.maturePartsLinkePosition=[]
    self.maturePartsLinkOrientation=[]

  def culc_coordinate(self,bodyId,cppn_outputs):
    #input coorinate
    #newInputId=p.getNumJoints(bodyId)
    NEWInputId=[len(self.matureLinkParentIndices)-2*(len(cppn_outputs)-(i+1)) for i in range(len((cppn_outputs)))]
    for newInputId in NEWInputId:
      parentPartsId=p.getJointInfo(bodyId,newInputId-1)[-1] 
      rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId]) #回転
      self.input_coordinate[newInputId]=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
      while parentPartsId!=-1:
        rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId])
        #self.input_coordinate[newInputId]+=rot.apply(np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]]))
        self.input_coordinate[newInputId]+=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
        parentPartsId=p.getJointInfo(bodyId,parentPartsId)[-1]

    #output coordinate
    #newOutputId=p.getNumJoints(bodyId)-1
    NEWOutputId=[len(self.matureLinkParentIndices)-2*(len(cppn_outputs)-(i+1))-1 for i in range(len((cppn_outputs)))]
    for newOutputId in NEWOutputId:
      parentPartsId=p.getJointInfo(bodyId,newOutputId-1)[-1]
      rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId])
      self.output_coordinate[newOutputId]=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
      while parentPartsId!=-1:
        rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId])
        # self.output_coordinate[newOutputId]+=rot.apply(np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]]))
        self.output_coordinate[newOutputId]+=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
        parentPartsId=p.getJointInfo(bodyId,parentPartsId)[-1]

  def get_jointPosition(self):
    for i in [j for j, key in self.identification.items() if key == 'box']:
      if self.jointGrobalPosition[i]==[]:
        jointLinkPosition=[[0,-2*self.HalfExtents[i][1]/3,0],[0,0,0],[0,2*self.HalfExtents[i][1]/3,0]]
        if i==0:
          jointGlobalPosition=jointLinkPosition
        else:
          rot= Rotation.from_quat(self.maturePartsLinkOrientation[i-1])
          jointGlobalPosition=[self.input_coordinate[i]+rot.apply(k) for k in jointLinkPosition]
        self.jointGrobalPosition[i]=jointGlobalPosition
        self.jointLinkPosition[i]=jointLinkPosition

  def create_base(self,sphereRadius,basePosition):
    self.sphereRadius = sphereRadius
    self.baseMass=1 #ベースとなる物体の質量
    self.baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[self.sphereRadius,3*self.sphereRadius,self.sphereRadius]) #ベースとなる物体(衝突判定有り)
    self.baseVisualShapeIndex =p.createVisualShape(p.GEOM_BOX, halfExtents=[self.sphereRadius,3*self.sphereRadius,self.sphereRadius],rgbaColor=[1,1,1,1])  #ブロックに色を指定する
    self.basePosition =basePosition #ベースの原点
    self.originPosition=basePosition
    self.baseOrientation = p.getQuaternionFromEuler([0,0,0]) #ベースの方向
    self.input_coordinate[0]=[0,0,0]
    self.partsType[0]=self.baseCollisionShapeIndex
    self.HalfExtents[0]=[self.sphereRadius,3*self.sphereRadius,self.sphereRadius]

    linkMasses=[] #linkしている物体の重量
    linkCollisionShapeIndices=[] #linkする物体(衝突判定有り)
    linkVisualShapeIndices=[] #linkする物体の外見
    linkPositions=[] #親link重心からから見た子link重心の位置
    linkOrientations=[] #linkの方向
    linkInertialFramePositions=[] #謎
    linkInertialFrameOrientations=[] #謎
    linkParentIndices=[] #親linkのid　id(int)は追加されたブロックから昇順
    linkJointTypes=[] #ジョイントの種類
    linkJointAxis=[] #ジョイントの軸
    self.identification[0]="box"
    for i in range(0):
      linkMasses.append(1)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=[self.sphereRadius,self.sphereRadius,self.sphereRadius]))
      linkVisualShapeIndices.append(-1)#p.createVisualShape(p.GEOM_BOX, halfExtents=[self.sphereRadius,self.sphereRadius,self.sphereRadius],rgbaColor=[1,1,1,1]))
      linkPositions.append([0,self.sphereRadius*2,0])
      # p.getQuaternionFromEuler(): quantanion を オイラーに変える
      linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
      linkInertialFramePositions=linkPositions
      linkInertialFrameOrientations=linkOrientations
      linkParentIndices.append(i)
      linkJointTypes.append(p.JOINT_REVOLUTE)
      linkJointAxis.append([0, 0, 1])    
      self.HalfExtents[i+1]=3*[self.sphereRadius]
      self.input_coordinate[i+1]=np.array(linkPositions[-1])+np.array(self.input_coordinate[i])
      self.output_coordinate[i+1]=self.input_coordinate[i+1]
      self.identification[i]="box"
    self.linkmasses=copy.copy(linkMasses)

    bodyId = p.createMultiBody( baseMass=self.baseMass,
                                baseCollisionShapeIndex = self.baseCollisionShapeIndex,
                                baseVisualShapeIndex = self.baseVisualShapeIndex,
                                basePosition = self.basePosition,
                                baseOrientation = self.baseOrientation,
                                linkMasses = linkMasses,
                                linkCollisionShapeIndices = linkCollisionShapeIndices,
                                linkVisualShapeIndices = linkVisualShapeIndices,
                                linkPositions= linkPositions,
                                linkOrientations= linkOrientations,
                                linkInertialFramePositions= linkInertialFramePositions,
                                linkInertialFrameOrientations= linkInertialFrameOrientations,
                                linkParentIndices= linkParentIndices,
                                linkJointTypes= linkJointTypes,
                                linkJointAxis= linkJointAxis)
    
    self.get_jointPosition()
    return bodyId

  def grow(self,bodyId,growstep,growRate,cppn_outputs):
    #cppn_outputs=[[scale,jointType,position,orientation,ParentInd]]

    r=0.1 #joint radius
    def create_joint():
      mass = r**2
      linkMasses.append(mass)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
      linkVisualShapeIndices.append(-1)
      #linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
      linkPositions.append(position)
      linkOrientations.append(orientation)
      linkParentIndices.append(parentInd)
      #linkJointTypes.append(p.JOINT_SPHERICAL)
      linkJointTypes.append(jointType)
      linkJointAxis.append([0, 0, 1])
      linkInertialFramePositions=linkPositions
      linkInertialFrameOrientations=linkOrientations
      
      self.immatureLinkPositions.append(position)
      self.immatureLinkOrientations.append(orientation)
      self.immatureLinkParentIndices.append(parentInd)

      self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_SPHERE,radius=r)
      self.jointsType[len(linkJointTypes)-1]=jointType

    self.basePosition=[*p.getBasePositionAndOrientation(bodyId)[0]]
    extraPartsNum=p.getNumJoints(bodyId) #発生イベント前の追加パーツの個数

    p.removeBody(bodyId)

    linkMasses=[] #linkしている物体の重量
    linkCollisionShapeIndices=[] #linkする物体(衝突判定有り)
    linkVisualShapeIndices=[] #linkする物体の外見
    linkPositions=[] #親link重心からから見た子link重心の位置
    linkOrientations=[] #linkの方向
    linkInertialFramePositions=[] #慣性　謎
    linkInertialFrameOrientations=[] #慣性　謎
    linkParentIndices=[] #親linkのid　id(int)は追加されたブロックから昇順
    linkJointTypes=[] #ジョイントの種類
    linkJointAxis=[] #ジョイントの軸

    if growstep==1:
      self.immatureLinkPositions=[]
      self.immatureLinkOrientations=[]
      self.immatureLinkParentIndices=[]

      for partsId in range(extraPartsNum):
        linkMasses.append(self.linkmasses[partsId])
        #linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId]))
        linkCollisionShapeIndices.append(self.partsType[partsId+1])
        # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId],rgbaColor=[1,1,1,1]))
        linkVisualShapeIndices.append(-1)
        linkPositions.append(self.maturePartsLinkePosition[partsId])
        linkOrientations.append(self.maturePartsLinkOrientation[partsId])
        linkParentIndices.append(self.matureLinkParentIndices[partsId])
        linkJointTypes.append(self.jointsType[partsId])
        linkJointAxis.append([0, 0, 1])   
      
      #新パーツ追加
      for i in range(len(cppn_outputs)):
        scale=cppn_outputs[i][0]
        jointType=cppn_outputs[i][1]
        position=cppn_outputs[i][2]
        orientation=cppn_outputs[i][3]
        parentInd=cppn_outputs[i][4]

        self.HalfExtents[extraPartsNum+2*(i+1)-1]=[r,r,r]
        self.HalfExtents[extraPartsNum+2*(i+1)]=[self.sphereRadius,self.sphereRadius*scale ,self.sphereRadius]
        newHalfExtents=[self.sphereRadius*growstep/growRate,self.sphereRadius*scale*growstep/growRate ,self.sphereRadius*growstep/growRate]
        newPartsPos=[0,self.HalfExtents[extraPartsNum+2*(i+1)-1][1]+self.sphereRadius*scale*growstep/growRate,0]

        create_joint()
        linkMasses.append(1*scale*growstep/growRate)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
        linkVisualShapeIndices.append(-1)
        #linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
        linkPositions.append(newPartsPos)
        linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
        linkParentIndices.append(extraPartsNum+2*(i+1)-1)
        linkJointTypes.append(p.JOINT_FIXED)
        linkJointAxis.append([0, 0, 1])

        self.immatureLinkPositions.append(linkPositions[-1])
        self.immatureLinkOrientations.append(linkOrientations[-1])
        self.immatureLinkParentIndices.append(linkParentIndices[-1])

        self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkCollisionShapeIndices)])
        self.jointsType[len(linkJointAxis)-1]=p.JOINT_FIXED
    
    else: #growstep!=1
      extraPartsNum=len(self.maturePartsLinkePosition) #成熟したパーツの数

      for partsId in range(extraPartsNum):
        linkMasses.append(self.linkmasses[partsId])
        linkCollisionShapeIndices.append(self.partsType[partsId+1])
        # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId],rgbaColor=[1,1,1,1]))
        linkVisualShapeIndices.append(-1)
        linkPositions.append(self.maturePartsLinkePosition[partsId])
        linkOrientations.append(self.maturePartsLinkOrientation[partsId])
        linkParentIndices.append(self.matureLinkParentIndices[partsId])
        linkJointTypes.append(self.jointsType[partsId])
        linkJointAxis.append([0, 0, 1])

      for i in range(len(cppn_outputs)):
        scale=cppn_outputs[i][0]
        jointType=cppn_outputs[i][1]
        newHalfExtents=[self.sphereRadius*growstep/growRate,self.sphereRadius*scale*growstep/growRate ,self.sphereRadius*growstep/growRate]
        newPartsPos=[0,self.HalfExtents[extraPartsNum+2*(i+1)-1][1]+self.sphereRadius*scale*growstep/growRate,0]
        #joint
        linkMasses.append(r**2)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
        linkVisualShapeIndices.append(-1)
        # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
        linkPositions.append(self.immatureLinkPositions[i*2])
        linkOrientations.append(self.immatureLinkOrientations[i*2])
        linkParentIndices.append(self.immatureLinkParentIndices[i*2])
        linkJointTypes.append(jointType)
        linkJointAxis.append([0, 0, 1])

        #box
        linkMasses.append(1*scale*growstep/growRate)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
        linkVisualShapeIndices.append(-1)
        # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
        linkPositions.append(newPartsPos)
        linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
        linkParentIndices.append(self.immatureLinkParentIndices[i*2+1])
        linkJointTypes.append(p.JOINT_FIXED)
        linkJointAxis.append([0, 0, 1])

    linkInertialFramePositions=linkPositions
    linkInertialFrameOrientations=linkOrientations

    # jointの最大数は127 ???
    bodyId = p.createMultiBody(baseMass=self.baseMass,
                                  baseCollisionShapeIndex = self.baseCollisionShapeIndex,
                                  baseVisualShapeIndex = self.baseVisualShapeIndex,
                                  basePosition = self.basePosition,
                                  baseOrientation = self.baseOrientation,
                                  linkMasses = linkMasses,
                                  linkCollisionShapeIndices = linkCollisionShapeIndices,
                                  linkVisualShapeIndices = linkVisualShapeIndices,
                                  linkPositions= linkPositions,
                                  linkOrientations= linkOrientations,
                                  linkInertialFramePositions= linkInertialFramePositions,
                                  linkInertialFrameOrientations= linkInertialFrameOrientations,
                                  linkParentIndices= linkParentIndices,
                                  linkJointTypes= linkJointTypes,
                                  linkJointAxis= linkJointAxis)
    
    if growstep==growRate:
      for i in range(len(cppn_outputs)):
        self.identification[extraPartsNum+2*(i+1)-1]="joint"
        self.identification[extraPartsNum+2*(i+1)]="box"
      self.matureLinkParentIndices=copy.copy(linkParentIndices)
      self.maturePartsLinkePosition=copy.deepcopy(linkPositions)
      self.maturePartsLinkOrientation=copy.deepcopy(linkOrientations)
      self.linkmasses=copy.copy(linkMasses)
      self.culc_coordinate(bodyId,cppn_outputs)
      self.get_jointPosition()

    return  bodyId

#ここからHyper-neat
# Config for CPPN.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_cppn_evo')
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            config_path)

sphereRadius=0.3

def eval_fitness(genomes,config):
  p.connect(p.GUI)
#   p.connect(p.DIRECT)
  p.createCollisionShape(p.GEOM_PLANE)
  p.createMultiBody(0, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  p.resetDebugVisualizerCamera(cameraDistance=50, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=[40,40,0])

  p.setGravity(0, 0, -10)
  p.setRealTimeSimulation(0)

  creatures={} 
  creatures["info"]=[] #各仮想生物の情報を保存
  creatures["id"]=[] #各仮想生物のIDを保存
  cppns=[]
  nets=[]
  ges=[]
  k=0
  ind=0
  for _,g in genomes:
    if np.mod(ind,10)==0:
      k+=1
    creatures["info"].append(creature())
    creatures["id"].append(creatures["info"][ind].create_base(sphereRadius,[10*np.mod(ind,10),10*k ,sphereRadius]))

    input_coordinates  = [(creatures["info"][ind].input_coordinate[i]) for i in creatures["info"][ind].input_coordinate]
    hidden_coordinates = [[(0.0, 0.0, 0)]]
    output_coordinates = [(creatures["info"][ind].output_coordinate[i]) for i in creatures["info"][ind].output_coordinate]
    sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

    cppns.append(neat.nn.FeedForwardNetwork.create(g, config))
    g.substrate=copy.deepcopy(sub)
    ges.append(g)
    nets.append(create_phenotype_network(cppns[ind], g.substrate))
    ind+=1

  growRate=100 #成長の刻み幅 growstep=growRate ならそのパーツの成長は終了
  growWaiting=500
  EventNum=3 #発生イベントの回数
  Total_Step=growWaiting*growRate*EventNum*2
  
  growstep=0 
  EventOppotunity=0
  anistropicFriction = [1, 0.01, 0.01]
  for step in tqdm(range(Total_Step)):
    # print(step)
    if np.mod(step,growWaiting)==0 and EventOppotunity<EventNum:
      growstep+=1
      if growstep==1:
        outputs=defaultdict(list)
        for ind, c in enumerate(creatures["info"]):
          input=list(c.input_coordinate.keys())
          for n in input:
            for m in [0,1,2]:
              if len(c.jointGrobalPosition[n][m])!=0:
                cppn_output=cppns[ind].activate(7*[0]+[step/Total_Step]+list(c.input_coordinate[n])+list(c.jointGrobalPosition[n][m]))
                cppn_addORnot=cppn_output[1]
                cppn_scale=cppn_output[2] if cppn_output[2]>4 else 4
                cppn_jointType=p.JOINT_REVOLUTE if cppn_output[3]>=0 else p.JOINT_FIXED
                cppn_orientation=p.getQuaternionFromEuler(cppn_output[4:])
                cppn_linkParentInd=n
                cppn_linkPositions=c.jointLinkPosition[n][m]
                c.jointGrobalPosition[n][m]=[]
                if cppn_addORnot>=0:
                  outputs[ind].append([cppn_scale,cppn_jointType,cppn_linkPositions,cppn_orientation,cppn_linkParentInd])
      for ind, c in enumerate(creatures["info"]):           
        if outputs[ind]!=[]:
          creatures["id"][ind]=c.grow(creatures["id"][ind],growstep,growRate,outputs[ind])#cppn_scale,cppn_jointType,cppn_linkPositions,cppn_orientation,cppn_linkParentInd)
          p.setCollisionFilterPair(creatures["id"][ind], creatures["id"][ind], -1,-1, enableCollision=0)
    if growstep==growRate:
      EventOppotunity+=1
      for ind, c in enumerate(creatures["info"]):
        growstep=0
        ges[ind].substrate.input_coordinates=[(c.input_coordinate[i]) for i in c.input_coordinate]
        ges[ind].substrate.output_coordinates=[(c.output_coordinate[i]) for i in c.output_coordinate]
        nets[ind]=create_phenotype_network(cppns[ind], ges[ind].substrate)

    for ind, c in enumerate(creatures["info"]):
      #関節を制限するらしい
      p.changeDynamics(creatures["id"][ind], -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
      p.getNumJoints(creatures["id"][ind])
      for i in range(p.getNumJoints(creatures["id"][ind])):
        p.changeDynamics(creatures["id"][ind], i, lateralFriction=2, anisotropicFriction=anistropicFriction) 

      jointlist=[i-1 for i, key in c.identification.items() if key == 'joint']
                                          
      p.setJointMotorControlArray(creatures["id"][ind],
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=nets[ind].activate([step/Total_Step] if len(jointlist)==0
                                    else [step/Total_Step]+[i[0] for i in p.getJointStates(creatures["id"][ind],jointlist)]),
                                  forces=[30]*len(jointlist))

    p.stepSimulation()
    #time.sleep(0.01)

  for ind, g in enumerate(ges):
    g.fitness=np.sqrt((np.array(creatures["info"][ind].originPosition)[0]-np.array(p.getBasePositionAndOrientation(creatures["id"][ind])[0][0]))**2
                        +(np.array(creatures["info"][ind].originPosition)[1]-np.array(p.getBasePositionAndOrientation(creatures["id"][ind])[0][1]))**2)
  p.disconnect()

def run(gens):
  pop = neat.population.Population(config)
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))

  winner = pop.run(eval_fitness, gens)
  print("hyperneat done")
  return winner, stats

# hoge= open("hyperneat_evocre_cppn.pkl", "rb")
# genome = pickle.load(hoge)
# print(type(genome))
# #eval_fitness(genome,config)

with open("hyperneat_evocre_cppn.pkl", "rb") as f:
    genome = pickle.load(f)
print(type(genome))
genomes = [(1, genome)]

eval_fitness(genomes,config)