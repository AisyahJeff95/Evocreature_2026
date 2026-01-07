import pybullet as p
import numpy as np
import copy
from scipy.spatial.transform import Rotation
from collections import defaultdict

class create:
  def __init__(self):
    self.HalfExtents={} #全ブロックの半径を格納
    self.input_coordinate={}
    self.output_coordinate={}
    self.partsType={}
    self.visualType={}
    self.jointsType={}
    self.identification={}
    self.jointGrobalPosition=defaultdict(list)
    self.jointLinkPosition={}
    self.linkmasses=[]
    self.matureLinkParentIndices=[]
    self.maturePartsLinkePosition=[]
    self.maturePartsLinkOrientation=[]
    self.partsColor=[0.8,0.8,0.8,1]

  def culc_coordinate(self,cppn_outputs):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.resetBasePositionAndOrientation(self.bodyId,[0,0,0],[0,0,0,1])
    jointlist=[i-1 for i, key in self.identification.items() if i-1!=-1]
    for jl in jointlist:
      p.resetJointState(self.bodyId,jl,0,0)
    parts_coordinate=[np.array([0,0,0])]+[np.array(i[0]) for i in p.getLinkStates(self.bodyId,self.matureLinkParentIndices)]
    for n,i in enumerate(parts_coordinate):
      try:
        self.output_coordinate[n]
        self.input_coordinate[n]
      except:
        if np.mod(n,2)==0:
          self.input_coordinate[n]=i
        if np.mod(n,2)==1:
          self.output_coordinate[n]=i
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # #input coorinate
    # #newInputId=p.getNumJoints(bodyId)
    # NEWInputId=[len(self.matureLinkParentIndices)-2*(len(cppn_outputs)-(i+1)) for i in range(len((cppn_outputs)))]
    # for newInputId in NEWInputId:
    #   rot = Rotation.from_quat(self.maturePartsLinkOrientation[newInputId-1]) #回転
    #   # self.input_coordinate[newInputId]=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId+1]))
    #   self.input_coordinate[newInputId]=rot.apply(np.array(self.maturePartsLinkePosition[newInputId-1]))
    #   parentPartsId=p.getJointInfo(self.bodyId,newInputId-1)[-1] 
    #   while parentPartsId!=-1:
    #     rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId])
    #     #self.input_coordinate[newInputId]+=rot.apply(np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]]))
    #     self.input_coordinate[newInputId]+=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
    #     parentPartsId=p.getJointInfo(self.bodyId,parentPartsId)[-1]

    # #output coordinate
    # #newOutputId=p.getNumJoints(bodyId)-1
    # NEWOutputId=[len(self.matureLinkParentIndices)-2*(len(cppn_outputs)-(i+1))-1 for i in range(len((cppn_outputs)))]
    # for newOutputId in NEWOutputId:
    #   rot = Rotation.from_quat(self.maturePartsLinkOrientation[newOutputId-1])
    #   # self.output_coordinate[newOutputId]=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId+1]))
    #   self.output_coordinate[newOutputId]=rot.apply(np.array(self.maturePartsLinkePosition[newOutputId-1]))
    #   parentPartsId=p.getJointInfo(self.bodyId,newOutputId-1)[-1]
    #   while parentPartsId!=-1:
    #     rot = Rotation.from_quat(self.maturePartsLinkOrientation[parentPartsId])
    #     # self.output_coordinate[newOutputId]+=rot.apply(np.array(linkPositions[p.getJointInfo(bodyId,parentPartsId)[-1]]))
    #     self.output_coordinate[newOutputId]+=rot.apply(np.array(self.maturePartsLinkePosition[parentPartsId]))
    #     parentPartsId=p.getJointInfo(self.bodyId,parentPartsId)[-1]

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

  def create_base(self,boxRadius,basePosition,baseOrientation,startPartsNum=1):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    self.boxRadius = boxRadius
    self.HalfExtents[0]=[self.boxRadius,4*self.boxRadius,self.boxRadius]
    self.baseMass=1 #ベースとなる物体の質量
    self.baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX,halfExtents=self.HalfExtents[0]) #ベースとなる物体(衝突判定有り)
    self.baseVisualShapeIndex =p.createVisualShape(p.GEOM_BOX, halfExtents=self.HalfExtents[0],rgbaColor=self.partsColor)  #ブロックに色を指定する
    self.basePosition =basePosition #ベースの原点
    self.baseOrientation = baseOrientation #ベースの方向
    self.input_coordinate[0]=np.array([0,0,0])
    self.partsType[0]=self.baseCollisionShapeIndex
    self.visualType[0]=self.baseVisualShapeIndex

    linkMasses=[] #linkしている物体の重量
    linkCollisionShapeIndices=[] #linkする物体(衝突判定有り)
    linkVisualShapeIndices=[] #linkする物体の外見
    linkPositions=[] #親link重心からから見た子link重心の位置
    linkOrientations=[] #linkの方向
    linkInertialFramePositions=[] ##ブロックの中心から重心へのベクトル
    linkInertialFrameOrientations=[] 
    linkParentIndices=[] #親linkのid　id(int)は追加されたブロックから昇順
    linkJointTypes=[] #ジョイントの種類
    linkJointAxis=[] #ジョイントの軸
    self.identification[0]="box"
    for i in range(startPartsNum-1):
      extraPartsNum=0
      r=copy.copy(self.boxRadius)
      self.HalfExtents[extraPartsNum+2*(i+1)-1]=[r,r,r]
      self.HalfExtents[extraPartsNum+2*(i+1)]=self.HalfExtents[0]

      # joint
      linkMasses.append(r**2)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
      # linkVisualShapeIndices.append(-1)
      linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_SPHERE, radius=r,rgbaColor=self.partsColor))
      linkPositions.append([0,self.HalfExtents[0][1]+r,0])
      linkOrientations.append([0,0,0,1])
      linkParentIndices.append(2*i)
      #linkJointTypes.append(p.JOINT_SPHERICAL)
      linkJointTypes.append(p.JOINT_REVOLUTE)
      linkJointAxis.append([0, 0, 1])
      self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_SPHERE,radius=r)
      self.visualType[len(linkVisualShapeIndices)]=p.createVisualShape(p.GEOM_SPHERE, radius=r,rgbaColor=self.partsColor)
      self.jointsType[len(linkJointTypes)-1]=linkJointTypes[-1]

      #box
      linkMasses.append(1)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=self.HalfExtents[extraPartsNum+2*(i+1)]))
      #linkVisualShapeIndices.append(-1)
      linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=self.HalfExtents[extraPartsNum+2*(i+1)],rgbaColor=self.partsColor))
      linkPositions.append([0,self.HalfExtents[extraPartsNum+2*(i+1)-1][1]+self.HalfExtents[extraPartsNum+2*(i+1)][1],0])
      linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
      linkParentIndices.append(extraPartsNum+2*(i+1)-1)
      linkJointTypes.append(p.JOINT_FIXED)
      linkJointAxis.append([0, 0, 1])
      self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkCollisionShapeIndices)])
      self.visualType[len(linkVisualShapeIndices)]=p.createVisualShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkCollisionShapeIndices)],rgbaColor=self.partsColor)
      self.jointsType[len(linkJointAxis)-1]=p.JOINT_FIXED

    linkInertialFramePositions=len(linkPositions)*[[0,0,0]]
    linkInertialFrameOrientations=linkOrientations
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
    
    for i in range(startPartsNum-1):
      self.identification[extraPartsNum+2*(i+1)-1]="joint"
      self.identification[extraPartsNum+2*(i+1)]="box"
    self.matureLinkParentIndices=copy.copy(linkParentIndices)
    self.maturePartsLinkePosition=copy.deepcopy(linkPositions)
    self.maturePartsLinkOrientation=copy.deepcopy(linkOrientations)
    self.linkmasses=copy.copy(linkMasses)
    self.bodyId=bodyId
    self.culc_coordinate((startPartsNum-1)*[[]])
    self.get_jointPosition()
    p.resetBasePositionAndOrientation(self.bodyId,basePosition,baseOrientation)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return bodyId
  
  def grow(self,growstep,growRate,cppn_outputs):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    #cppn_outputs=[[scale,jointType,position,orientation,ParentInd]]

    r=copy.copy(self.boxRadius) #joint radius
    def create_joint():
      mass = 0
      linkMasses.append(mass)
      linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
      # linkVisualShapeIndices.append(-1)
      linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_SPHERE, radius=r,rgbaColor=self.partsColor))
      linkPositions.append(position)
      linkOrientations.append(orientation)
      linkParentIndices.append(parentInd)
      #linkJointTypes.append(p.JOINT_SPHERICAL)
      linkJointTypes.append(jointType)
      linkJointAxis.append([0, 0, 1])
      # linkInertialFramePositions=linkPositions
      # linkInertialFrameOrientations=linkOrientations
      
      self.immatureLinkPositions.append(position)
      self.immatureLinkOrientations.append(orientation)
      self.immatureLinkParentIndices.append(parentInd)

      self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_SPHERE,radius=r)
      self.visualType[len(linkVisualShapeIndices)]=p.createVisualShape(p.GEOM_SPHERE, radius=r,rgbaColor=self.partsColor)
      self.jointsType[len(linkJointTypes)-1]=jointType

    # 発生イベント前の関節の角度と体の向きを保存
    basePO=list(p.getBasePositionAndOrientation(self.bodyId))
    # basePO=(tuple(np.array(basePO[0])+np.array([0,0,(self.boxRadius*growRate)/growRate])),basePO[1]) #跳ね軽減
    baseVelocity=p.getBaseVelocity(self.bodyId)

    jointlist=[i-1 for i, key in self.identification.items() if i-1!=-1]
    if jointlist!=[]:
      JointPosition=[i[0] for i in p.getJointStates(self.bodyId,jointlist)]
      JointVelocity=[i[1] for i in p.getJointStates(self.bodyId,jointlist)]

    self.basePosition=basePO[0]
    extraPartsNum=p.getNumJoints(self.bodyId) #発生イベント前の追加パーツの個数

    p.removeBody(self.bodyId)

    linkMasses=[] #linkしている物体の重量
    linkCollisionShapeIndices=[] #linkする物体(衝突判定有り)
    linkVisualShapeIndices=[] #linkする物体の外見
    linkPositions=[] #親link重心からから見た子link重心の位置
    linkOrientations=[] #linkの方向
    linkInertialFramePositions=[] #ブロックの中心から重心へのベクトル
    linkInertialFrameOrientations=[] #ブロックの中心から重心への方向
    linkParentIndices=[] #親linkのid　id(int)は追加されたブロックから昇順
    linkJointTypes=[] #ジョイントの種類
    linkJointAxis=[] #ジョイントの軸

    if growstep==1:
      self.immatureLinkPositions=[]
      self.immatureLinkOrientations=[]
      self.immatureLinkParentIndices=[]

      for partsId in range(extraPartsNum):
        linkMasses.append(self.linkmasses[partsId])
        linkCollisionShapeIndices.append(self.partsType[partsId+1])
        linkVisualShapeIndices.append(self.visualType[partsId+1])
        # linkVisualShapeIndices.append(-1)
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
        self.HalfExtents[extraPartsNum+2*(i+1)]=[self.boxRadius,self.boxRadius*scale ,self.boxRadius]
        newHalfExtents=[self.boxRadius*growstep/growRate,self.boxRadius*scale*growstep/growRate ,self.boxRadius*growstep/growRate]
        newPartsPos=[0,self.HalfExtents[extraPartsNum+2*(i+1)-1][1]+self.boxRadius*scale*growstep/growRate,0]

        create_joint()
        linkMasses.append(self.boxRadius**3*scale*growstep/growRate)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
        # linkVisualShapeIndices.append(-1)
        linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=self.partsColor))
        linkPositions.append(newPartsPos)
        linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
        linkParentIndices.append(extraPartsNum+2*(i+1)-1)
        linkJointTypes.append(p.JOINT_FIXED)
        linkJointAxis.append([0, 0, 1])

        self.immatureLinkPositions.append(linkPositions[-1])
        self.immatureLinkOrientations.append(linkOrientations[-1])
        self.immatureLinkParentIndices.append(linkParentIndices[-1])

        self.partsType[len(linkCollisionShapeIndices)]=p.createCollisionShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkCollisionShapeIndices)])
        self.visualType[len(linkVisualShapeIndices)]=p.createVisualShape(p.GEOM_BOX, halfExtents=self.HalfExtents[len(linkCollisionShapeIndices)],rgbaColor=self.partsColor)
        self.jointsType[len(linkJointAxis)-1]=p.JOINT_FIXED
    
    else: #growstep!=1
      extraPartsNum=len(self.maturePartsLinkePosition) #成熟したパーツの数

      for partsId in range(extraPartsNum):
        linkMasses.append(self.linkmasses[partsId])
        linkCollisionShapeIndices.append(self.partsType[partsId+1])
        linkVisualShapeIndices.append(self.visualType[partsId+1])
        # linkVisualShapeIndices.append(-1)
        linkPositions.append(self.maturePartsLinkePosition[partsId])
        linkOrientations.append(self.maturePartsLinkOrientation[partsId])
        linkParentIndices.append(self.matureLinkParentIndices[partsId])
        linkJointTypes.append(self.jointsType[partsId])
        linkJointAxis.append([0, 0, 1])

      for i in range(len(cppn_outputs)):
        scale=cppn_outputs[i][0]
        jointType=cppn_outputs[i][1]
        newHalfExtents=[self.boxRadius*growstep/growRate,self.boxRadius*scale*growstep/growRate ,self.boxRadius*growstep/growRate]
        newPartsPos=[0,self.HalfExtents[extraPartsNum+2*(i+1)-1][1]+self.boxRadius*scale*growstep/growRate,0]
        #joint
        linkMasses.append(0)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_SPHERE,radius=r))
        # linkVisualShapeIndices.append(-1)
        linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_SPHERE, radius=r ,rgbaColor=self.partsColor))
        linkPositions.append(self.immatureLinkPositions[i*2])
        linkOrientations.append(self.immatureLinkOrientations[i*2])
        linkParentIndices.append(self.immatureLinkParentIndices[i*2])
        linkJointTypes.append(jointType)
        linkJointAxis.append([0, 0, 1])

        #box
        linkMasses.append(self.boxRadius**3*scale*growstep/growRate)
        linkCollisionShapeIndices.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=newHalfExtents))
        # linkVisualShapeIndices.append(-1)
        linkVisualShapeIndices.append(p.createVisualShape(p.GEOM_BOX, halfExtents=newHalfExtents,rgbaColor=self.partsColor))
        linkPositions.append(newPartsPos)
        linkOrientations.append([*p.getQuaternionFromEuler([0,0,0])])
        linkParentIndices.append(self.immatureLinkParentIndices[i*2+1])
        linkJointTypes.append(p.JOINT_FIXED)
        linkJointAxis.append([0, 0, 1])

    linkInertialFramePositions=len(linkPositions)*[[0,0,0]]
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
                                  # flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
    
    if growstep==growRate:
      for i in range(len(cppn_outputs)):
        self.identification[extraPartsNum+2*(i+1)-1]="joint"
        self.identification[extraPartsNum+2*(i+1)]="box"
      self.matureLinkParentIndices=copy.copy(linkParentIndices)
      self.maturePartsLinkePosition=copy.deepcopy(linkPositions)
      self.maturePartsLinkOrientation=copy.deepcopy(linkOrientations)
      self.linkmasses=copy.copy(linkMasses)
      self.culc_coordinate(cppn_outputs)
      self.get_jointPosition()

    # 発生イベント前の体の向きと関節の角度を適応
    if jointlist!=[]:
      for jl,js,jv in zip(jointlist,JointPosition,JointVelocity):
        p.resetJointState(self.bodyId,jl,js,jv)
    p.resetBasePositionAndOrientation(self.bodyId,basePO[0],basePO[1])
    p.resetBaseVelocity(self.bodyId,baseVelocity[0],baseVelocity[1])

    self.bodyId=bodyId

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return  bodyId
  
  def eye(self,width,height):
    fov = 60
    aspect = width / height
    near = 0.01
    far = 100
    # rot= Rotation.from_quat(p.getBasePositionAndOrientation(self.bodyId)[1])
    Euler=list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.bodyId)[1]))
    Euler[0],Euler[1]=0,0
    rot= Rotation.from_quat(p.getQuaternionFromEuler(Euler))

    cameraPos =np.array(p.getBasePositionAndOrientation(self.bodyId)[0])+np.array([self.HalfExtents[0][0],0,6])
    camTargetPos =rot.apply([-100,0,50])
    cameraUp = [0, 0, 1]

    view_matrix=p.computeViewMatrix(cameraPos,camTargetPos,cameraUp)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    images = p.getCameraImage(width,
                              height,
                              view_matrix,
                              projection_matrix,
                              shadow=False,
                              renderer=p.ER_TINY_RENDERER)
    rgb_image = np.reshape(images[2], (height, width, 4))/255
    gray_image = 0.299 * rgb_image[:, :, 2] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 0]
    return gray_image
  
  def radar_1(self,radarAccuracy,detectionRange,targetId):
    radar=[]
    Euler=list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.bodyId)[1]))
    Euler[0],Euler[1]=0,0
    rot= Rotation.from_quat(p.getQuaternionFromEuler(Euler))
    for i in range(radarAccuracy):
      detection=p.rayTest(p.getBasePositionAndOrientation(self.bodyId)[0],rot.apply([detectionRange*np.cos((i/radarAccuracy)*2*np.pi),detectionRange*np.sin((i/radarAccuracy)*2*np.pi),2*self.boxRadius]))
      detectionId=detection[0][0]
      if detectionId==targetId:
        radar.append(1)
      elif detectionId==self.bodyId or detectionId==-1:
        radar.append(0)
      else:
        radar.append(-1)
    return radar

  def radar_2(self,radarAccuracy,detectionRange,targetId):
    radar=[]
    Euler=list(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.bodyId)[1]))
    Euler[0],Euler[1]=0,0
    rot= Rotation.from_quat(p.getQuaternionFromEuler(Euler))
    for i in range(radarAccuracy):
      detection=p.rayTest(p.getBasePositionAndOrientation(self.bodyId)[0],rot.apply([detectionRange*np.cos((i/radarAccuracy)*2*np.pi),detectionRange*np.sin((i/radarAccuracy)*2*np.pi),2*self.boxRadius]))
      detectionDistance=np.linalg.norm(np.array(detection[0][3])-np.array(p.getBasePositionAndOrientation(self.bodyId)[0]))
      detectionId=detection[0][0]
      if detectionId==targetId:
        radar.append(detectionDistance)
      elif detectionId==self.bodyId or detectionId==-1:
        radar.append(0)
      else:
        radar.append(-1*detectionDistance)
    return radar

  def radar(self,targetId):
    radar=[0]*8

    selfPositionAndOrientation=p.getBasePositionAndOrientation(self.bodyId)
    selfPosition=np.array(selfPositionAndOrientation[0])
    selfOrientation=np.array(selfPositionAndOrientation[1])

    targetPositionAndOrientation=p.getBasePositionAndOrientation(targetId)
    targetPosition=np.array(targetPositionAndOrientation[0])
    targetOrientation=np.array(targetPositionAndOrientation[1])

    # 自身と他個体の距離
    distance=np.sqrt((selfPosition[0]-targetPosition[0])**2+(selfPosition[1]-selfPosition[1])**2)
    # 自身の絶対角度
    self_angle=p.getEulerFromQuaternion(selfOrientation)[-1]
    # 自身と相手の相対角度
    self_another_angle=np.arctan2(targetPosition[1]-selfPosition[1],targetPosition[0]-selfPosition[0])
    # 自身の向いてる方向からの相手の方向
    total_angle=np.mod(2*np.pi+self_another_angle-self_angle,2*np.pi)
    # 音が聞こえてくる方向インデックス
    orientation_index=np.array(total_angle//(2*np.pi/8),dtype=int)

    radar[orientation_index]=distance

    return radar
