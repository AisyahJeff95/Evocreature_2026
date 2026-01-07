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

class creature:
  def __init__(self,sphereRadius):
    self.sphereRadius = sphereRadius
    self.HalfExtents={} #全ブロックの半径を格納
    self.HalfExtents[0]=3*[self.sphereRadius]
    self.input_coordinate={}
    self.output_coordinate={}
    self.partsType={}
    self.jointsType={}
    self.identification={}
  def culc_coordinate(self,bodyId,linkPositions):
    #input coorinate
    newInputId=p.getNumJoints(bodyId)
    parentPartsId=p.getJointInfo(bodyId,p.getNumJoints(bodyId)-1)[-1]   
    self.input_coordinate[newInputId]=np.array(linkPositions[newInputId-1])
    while parentPartsId!=-1:
      self.input_coordinate[newInputId]+=np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]])
      parentPartsId=p.getJointInfo(bodyId,parentPartsId)[-1]

    #output coordinate
    newOutputId=p.getNumJoints(bodyId)-1
    parentPartsId=p.getJointInfo(bodyId,p.getNumJoints(bodyId)-2)[-1]
    self.output_coordinate[newOutputId]=np.array(linkPositions[newOutputId-1])
    while parentPartsId!=-1:
      self.output_coordinate[newOutputId]+=np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]])
      parentPartsId=p.getJointInfo(bodyId,parentPartsId)[-1]

  def create_base(self):

    self.baseMass=1 #ベースとなる物体の質量
    self.baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=[self.sphereRadius,self.sphereRadius,self.sphereRadius]) #ベースとなる物体(衝突判定有り)
    self.baseVisualShapeIndex =p.createVisualShape(p.GEOM_BOX, halfExtents=[self.sphereRadius,self.sphereRadius,self.sphereRadius],rgbaColor=[1,1,1,1])  #ブロックに色を指定する
    self.basePosition = [0, 0,self.sphereRadius] #ベースの原点
    self.baseOrientation = p.getQuaternionFromEuler([0,0,0]) #ベースの方向
    self.input_coordinate[0]=self.basePosition
    self.partsType[0]=self.baseCollisionShapeIndex

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
    self.matureLinkParentIndices=linkParentIndices
    self.maturePartsLinkePosition=linkPositions
    self.maturePartsLinkOrientation=linkOrientations

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
                    
    return bodyId

  def grow(self,bodyId,scale,growstep,growRate):
    
    r=0.1 #joint radius
    def create_joint():
      mass = r
      linkMasses.append(mass)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
      #linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX,halfExtents=3*[r]))
      linkVisualShapeIndices.append(-1)
      #linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
      linkPositions.append([0,self.HalfExtents[extraPartsNum][1]+r,0])
      linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
      linkParentIndices.append(extraPartsNum)
      #linkJointTypes.append(p.JOINT_SPHERICAL)
      linkJointTypes.append(p.JOINT_REVOLUTE)
      linkJointAxis.append([0, 0, 1])
      linkInertialFramePositions=linkPositions
      linkInertialFrameOrientations=linkOrientations

      self.partsType[len(linkParentIndices)]=p.createCollisionShape(p.GEOM_SPHERE,radius=r)
      #self.partsType[len(linkParentIndices)]=p.createCollisionShape(p.GEOM_BOX,halfExtents=3*[r])
      self.jointsType[len(linkJointTypes)-1]=p.JOINT_REVOLUTE

    self.basePosition=[*p.getBasePositionAndOrientation(bodyId)[0]]
    extraPartsNum=p.getNumJoints(bodyId)


    p.removeBody(bodyId)

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

    newHalfExtents=[self.sphereRadius,self.sphereRadius*scale*growstep/growRate ,self.sphereRadius]

    if growstep==1:
      self.identification[extraPartsNum+1]="joint"
      self.identification[extraPartsNum+2]="box"

      self.HalfExtents[extraPartsNum+1]=[r,r,r]
      self.HalfExtents[extraPartsNum+2]=[self.sphereRadius,self.sphereRadius*scale ,self.sphereRadius]
      newPartsPos=[0,self.HalfExtents[extraPartsNum][1]+self.sphereRadius*scale*growstep/growRate,0]
      
      for partsId in range(extraPartsNum):
        linkMasses.append(1)
        #linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId]))
        linkCollisionShapeIndices.append(self.partsType[partsId+1])
        # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId],rgbaColor=[1,1,1,1]))
        linkVisualShapeIndices.append(-1)
        linkPositions.append(self.maturePartsLinkePosition[partsId])
        linkOrientations.append(self.maturePartsLinkOrientation[partsId])
        linkParentIndices.append(partsId)
        linkJointTypes.append(self.jointsType[partsId])
        linkJointAxis.append([0, 0, 1])   

      #新パーツ追加
      create_joint()
      self.maturePartsLinkePosition.append([0,self.HalfExtents[extraPartsNum][1]+r,0])
      self.maturePartsLinkOrientation.append([0,0,0,1])
      linkMasses.append(1*scale*growstep/growRate)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
      linkVisualShapeIndices.append(-1)
      #linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
      linkPositions.append(newPartsPos)
      linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
      linkParentIndices.append(extraPartsNum+1)
      linkJointTypes.append(p.JOINT_FIXED)
      linkJointAxis.append([0, 0, 1])

      self.partsType[len(linkParentIndices)]=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkParentIndices)])
      self.jointsType[len(linkJointAxis)-1]=p.JOINT_FIXED
    else:
      newPartsPos=[0,self.HalfExtents[extraPartsNum-1][1]+self.sphereRadius*scale*growstep/growRate,0]
      
      # linkCollisionShapeIndices=[self.partsType[i] for i in range(1,len(self.partsType)-1)]
      # linkPositions=copy.deepcopy(self.maturePartsLinkePosition)
      # linkOrientations=copy.deepcopy(self.maturePartsLinkOrientation)
      # linkJointTypes=[self.jointsType[i] for i in range(0,len(self.jointsType))]

      for partsId in range(extraPartsNum):
        if partsId<extraPartsNum-1:
          linkMasses.append(1)
          linkCollisionShapeIndices.append(self.partsType[partsId+1])
          # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=HalfExtents[partsId],rgbaColor=[1,1,1,1]))
          linkVisualShapeIndices.append(-1)
          linkPositions.append(self.maturePartsLinkePosition[partsId])
          linkOrientations.append(self.maturePartsLinkOrientation[partsId])
        else: 
          linkMasses.append(1*scale*growstep/growRate)
          linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
          linkVisualShapeIndices.append(-1)
          # linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=[1,1,1,1]))
          linkPositions.append(newPartsPos)
          linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
        
        linkParentIndices.append(partsId)
        linkJointTypes.append(self.jointsType[partsId])
        linkJointAxis.append([0, 0, 1])

    linkInertialFramePositions=linkPositions
    linkInertialFrameOrientations=linkOrientations
  
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
      self.culc_coordinate(bodyId,linkPositions)
      self.matureLinkParentIndices=linkParentIndices
      self.maturePartsLinkePosition=copy.deepcopy(linkPositions)
      self.maturePartsLinkOrientation=copy.deepcopy(linkOrientations)

    return  bodyId


#ここからHyper-neat
file_name="config_cppn_evo"
# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            file_name)

sphereRadius=0.3

def eval_fitness(genomes,config):
  p.connect(p.GUI)
  p.createCollisionShape(p.GEOM_PLANE)
  p.createMultiBody(0, 0)
  p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
  c=creature(sphereRadius)
  sphereUid=c.create_base()

  input_coordinates  = [(c.input_coordinate[i]) for i in c.input_coordinate]
  hidden_coordinates = [[(0.0, 0.0, 0)]]
  output_coordinates = [(c.output_coordinate[i]) for i in c.output_coordinate]

  sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)



  p.setGravity(0, 0, -10)
  p.setRealTimeSimulation(0)

  for _,g in genomes:
    cppn=neat.nn.FeedForwardNetwork.create(g, config)
    g.substrate=copy.deepcopy(sub)
    net = create_phenotype_network(cppn, g.substrate)

    step=0
    growstep=0
    growRate=10
    ExtraPartsThersh=3
    ExtraParts=0

    anistropicFriction = [1, 0.01, 0.01]

    while (p.isConnected()):
      step+=1
      print(step)
      if np.mod(step,300)==0 and ExtraParts<ExtraPartsThersh*2:
        step+=1
        for i in range(growRate):
          growstep+=1
          jointlist=[i-1 for i, key in c.identification.items() if key == 'joint']
          if jointlist!=[]:
            jointstate=[i[0] for i in p.getJointStates(sphereUid,jointlist)]
          sphereUid=c.grow(sphereUid,2,growstep,growRate)
          if jointlist!=[]:
            for i,j in zip(jointlist,jointstate):
              p.resetJointState(sphereUid,i,j)
          # p.stepSimulation()
          # time.sleep(0.01)

          if growstep==growRate:
            ExtraParts+=1
            growstep=0
            g.substrate.input_coordinates=[(c.input_coordinate[i]) for i in c.input_coordinate]
            g.substrate.output_coordinates=[(c.output_coordinate[i]) for i in c.output_coordinate]
            net = create_phenotype_network(cppn, g.substrate)

      #関節を制限するらしい
      p.changeDynamics(sphereUid, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
      p.getNumJoints(sphereUid)
      for i in range(p.getNumJoints(sphereUid)):
        p.changeDynamics(sphereUid, i, lateralFriction=2, anisotropicFriction=anistropicFriction) 

      jointlist=[i-1 for i, key in c.identification.items() if key == 'joint']
      p.setJointMotorControlArray(sphereUid,
                                  jointlist,
                                  p.POSITION_CONTROL,
                                  targetPositions=net.activate([np.mod(step,100)] if jointlist==[] 
                                                                else [np.mod(step,100)]+[i[0] for i in p.getJointStates(sphereUid,jointlist)]),
                                  forces=[1]*len(jointlist))

      p.stepSimulation()
      time.sleep(0.01)
      
      if step>2000:
        fitness=100*np.sqrt((p.getBasePositionAndOrientation(sphereUid)[0][0]+p.getBasePositionAndOrientation(sphereUid)[0][1])**2)
        print("【fitness is "+str(fitness)+"】")
        #p.disconnect()
    g.fitness=fitness

def run(gens):
  pop = neat.population.Population(config)
  stats = neat.statistics.StatisticsReporter()
  pop.add_reporter(stats)
  pop.add_reporter(neat.reporting.StdOutReporter(True))

  winner = pop.run(eval_fitness, gens)
  print("hyperneat done")
  return winner, stats

if __name__ == '__main__':
    winner = run(1)[0]