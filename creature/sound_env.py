import pybullet as p
import numpy as np
from collections import defaultdict
from pureples.shared.substrate import Substrate
from pureples.hyperneat import create_phenotype_network

class Sound():
  def __init__(self,genomes,boxRadius):
    self.spring_k=100
    self.frequency_min=1/(2*np.pi)*np.sqrt(self.spring_k/(boxRadius*8)**3*2)
    self.frequency_max=1/(2*np.pi)*np.sqrt(self.spring_k/(boxRadius)**3*2)
    self.frequency_range=np.linspace(self.frequency_min,self.frequency_max,5)
    self.hear_orientation_range=np.linspace(0, 2*np.pi, 8)
    
    # self.sound_matrix=[[方向0,周波数0],[方向0,周波数1],[方向0,周波数2]...]
    self.sound_matrix=[[0.5*i*boxRadius*np.cos(2*np.pi*((1/16)+j/len(self.hear_orientation_range))),0.5*i*boxRadius*np.sin(2*np.pi*((1/16)+j/len(self.hear_orientation_range))),0] for j in range(len(self.hear_orientation_range)) for i in range(1,len(self.frequency_range))]

    for _,g in genomes:

      input_coordinates  = self.sound_matrix+[(g.creature.input_coordinate[i]) for i in g.creature.input_coordinate]   
      hidden_coordinates = [[(i, 0.0, 1) for i in range(-4,5)]]
      output_coordinates = [(g.creature.output_coordinate[i]) for i in g.creature.output_coordinate]

      g.substrate=Substrate(input_coordinates, output_coordinates, hidden_coordinates)
      g.net=create_phenotype_network(g.cppn, g.substrate)


  def create_substrate_network(self,g):
    g.substrate.input_coordinates=self.sound_matrix+[c for _,c in g.creature.input_coordinate.items()]
    g.substrate.output_coordinates=[c for _,c in g.creature.output_coordinate.items()]
    g.net=create_phenotype_network(g.cppn, g.substrate)


  def listening(self,genomes):
    self.k=5 # 音の減衰率
    Sounds=[]

    for _,g in genomes:
      g.creature.sound_volume_of=np.zeros([len(self.hear_orientation_range),len(self.frequency_range)-1])

      boxlist=[i-1 for i, key in g.creature.identification.items() if key == 'box']
      contactlist=[]
      if len(boxlist)>=2:
        contacts=[p.getContactPoints(g.creature.bodyId,g.creature.bodyId,boxlist[i],boxlist[j]) for i in range(len(boxlist)) for j in range(i+1,len(boxlist))]
        for c in contacts: 
          if len(c)>0 and c[0][8]<0:
            contactlist.append((c[0][3]+1,c[0][4]+1))

      Sounds.append(np.zeros(len(self.frequency_range)-1))
      
      if contactlist!=[]:

        jointlist=[i-1 for i, key in g.creature.identification.items() if key == 'joint']
        if jointlist!=[]:
          parts_velocity={0:0} #base parts
          # j+2: link_ind→parts_ind +1, joint→box +1
          parts_velocity.update({j+2:p.getJointState(g.creature.bodyId,j)[1] for j in jointlist})
        parts_mass={b+1: p.getDynamicsInfo(g.creature.bodyId,b)[0] for b in boxlist}
        
        #jointID=boxID-1
        sound_volume=np.array([np.abs(parts_velocity[c[0]]-parts_velocity[c[1]])*(parts_mass[c[0]]+parts_mass[c[1]]) for c in contactlist])
        sound_frequency=np.array([(1/(2*np.pi))*np.sqrt(self.spring_k/(parts_mass[c[0]]+parts_mass[c[1]])) for c in contactlist])

        frecuency_index=np.asarray(sound_frequency//((self.frequency_max-self.frequency_min)/len(self.frequency_range)), dtype=int)
        Sounds[-1][frecuency_index]+=sound_volume

    PositionsAndOrientations=[p.getBasePositionAndOrientation(g.creature.bodyId) for _,g in genomes]
    positions=np.array([p[0] for p in PositionsAndOrientations])

    # 自身と他個体の距離
    distance=np.sqrt(np.array([positions[:,0]-p[0] for p in positions])**2 +np.array([positions[:,1]-p[1] for p in positions])**2)
    # 自身の絶対角度
    self_angle=np.array([p.getEulerFromQuaternion(o[1])[-1] for o in PositionsAndOrientations])
    # 自身と相手の相対角度
    self_another_angle=np.arctan2([positions[:,0]-p[0] for p in positions],[positions[:,1]-p[1] for p in positions])
    # 自身の向いてる方向からの相手の方向
    total_angle=np.array([np.mod(2*np.pi+self_another_angle[i]-self_angle[i],2*np.pi) for i in range(len(self_angle))])
    # 音が聞こえてくる方向インデックス
    orientation_index=np.asarray(total_angle//(2*np.pi/len(self.hear_orientation_range)),dtype=int)

    # 距離と方向の行列から自身のインデックスを除外
    self_index=np.ones([len(orientation_index),len(orientation_index)],dtype=bool)
    self_index[np.diag_indices(len(genomes))] = False
    orientation_index=np.reshape(orientation_index[self_index],[len(orientation_index),len(orientation_index)-1])
    distance=np.reshape(distance[self_index],[len(distance),len(distance)-1])

    for n,(_,g) in enumerate(genomes):
      # orientation_index[n] インデックスn の個体から見た他個体の方向インデックス行列
      # distance[n] インデックスn の個体から見た他個体の距離行列
      # Sounds インデックスn の個体が鳴らす周波数ごとの音量
      g.creature.sound_volume_of[orientation_index[n]]+=np.transpose(1/(np.e**np.array([distance[n]])/self.k)*np.transpose([s for m,s in enumerate(Sounds) if m!=n]))
      
      # for o,i in enumerate(orientation_index[n]):
      #   g.creature.sound_volume_of[i]+=np.transpose(1/(np.e**np.array([distance[n]])/self.k)*np.transpose([s for m,s in enumerate(Sounds) if m!=n]))[o]

    return Sounds