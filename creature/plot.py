from matplotlib import pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
from pylab import rcParams
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle


def trajectory(pop_radius,genomes):
  #rcParams['figure.figsize'] = 15,15

  #円の設定
  c = patches.Circle(xy=(0, 0), radius=pop_radius, ec='black', fill=False)
  x1 = patches.Circle(xy=(0, 0), radius=2.5, ec='orange', fill=False)
  # x2 = patches.Rectangle(xy=(-60,-60), width=120, height=120, angle=0.0, color='black', fill=False) #outline valley
  # x3 = patches.Circle(xy=(0, 20), radius=12, ec='orange', fill=False) #inline valley
  #x4 = patches.Circle(xy=(0, -20), radius=12, ec='orange', fill=False)
  ax = plt.axes()
  ax.add_patch(c)
  ax.add_patch(x1)
  # ax.add_patch(x2)  
  # ax.add_patch(x3)
  #ax.add_patch(x4)
  #足跡
  for n,i in enumerate(genomes):
    plt.scatter([j[0] for m,j in enumerate(i[1].position) if np.mod(m,10)==0],[j[1] for m,j in enumerate(i[1].position) if np.mod(m,10)==0], s=2.0, color=cm.rainbow(float(n) / len(genomes))) 
    # plt.xlim(50.0, -50.0) # (3)x軸の表示範囲
    # plt.ylim(-50.0, 50.0) # (4)y軸の表示範囲
    plt.xlabel('x direction') # x軸
    plt.ylabel('y direction') # y軸
    # plt.axis('scaled')
    # ax.set_xlim(-100,100)
    # ax.set_ylim(-100,100)
    ax.set_aspect('equal')
    # figureの保存
  plt.savefig(os.path.dirname(__file__) + "/trajectory.png")
  # plt.close()

def trajectory_2(best_genome,data,radius):
  #rcParams['figure.figsize'] = 15,15

  #円の設定
  c = patches.Circle(xy=(0, 0), radius=radius, ec='black', fill=False)
  x1 = patches.Circle(xy=(0, 0), radius=2.5, ec='orange', fill=False)
  x2 = patches.Rectangle(xy=(-80,-80), width=160, height=160, angle=0.0, color='black', fill=False) #outline valley
  x3 = patches.Circle(xy=(0, 20), radius=12, ec='orange', fill=False) #inline valley
  #x4 = patches.Circle(xy=(0, -20), radius=12, ec='orange', fill=False)
  ax = plt.axes()
  ax.add_patch(c)
  ax.add_patch(x1)
  ax.add_patch(x2)
  ax.add_patch(x3)
  #ax.add_patch(x4)
  #足跡
  g=best_genome[1]
  for n,i in enumerate(data[best_genome[0]]):
    if np.mod(n,10)==0:
      sc=plt.scatter(i[0],i[1], s=2.0, vmin=g.partsNum[0],vmax=g.partsNum[-1],c=g.partsNum[n], cmap=cm.rainbow) # １列目のデータをx軸の値、3列目のデータをy軸の値として与える。
      # plt.xlim(50.0, -50.0) # (3)x軸の表示範囲
      # plt.ylim(-50.0, 50.0) # (4)y軸の表示範囲
  plt.xlabel('x direction') # x軸
  plt.ylabel('y direction') # y軸
  plt.axis('scaled')
  plt.colorbar(sc)
  # ax.set_xlim(42,-42)
  # ax.set_ylim(-42,42)
  ax.set_aspect('equal')
  # figureの保存
  # plt.savefig(os.path.dirname(__file__) + "/trajectory_2.png")
  # plt.close()


def partsNum(Genomes,mode=None):
  total_fitness=[]
  parts={}
  best=[]
  for n,genome in enumerate(Genomes):
      b=0
      gen_fitness=[]
      parts[n]=[]
      for ind,g in genome:
          parts[n].append((ind,len(g.substrate.output_coordinates)+1))
          gen_fitness.append(g.fitness)
          if b<g.fitness:
              b=g.fitness
              b_ind=ind
      best.append(b_ind)
      gen_fitness.sort()
      total_fitness.append(gen_fitness)
  
  Ave_parts=[]
  Best_parts=[]
  Most_parts=[]
  for ind,p in parts.items():
    ave_parts=[]
    best_parts=[]
    most_parts=[]
    ave_parts=np.mean(([q[1] for q in p]))
    best_parts=[q[1] for q in p if q[0]==best[ind]]
    most_parts=sorted([(q[1],q[0]) for q in p])[-1]
    Ave_parts.append(ave_parts)
    Best_parts.append(*best_parts)
    Most_parts.append(most_parts)

  plt.figure()
  if mode=="most":
    plt.plot([gen for gen in range(len(Genomes))],Ave_parts,[gen for gen in range(len(Genomes))],Best_parts,[gen for gen in range(len(Genomes))],[M[0] for M in Most_parts])
    plt.legend(["Average","Best","Most"])
  else:
    plt.plot([gen for gen in range(len(Genomes))],Ave_parts,[gen for gen in range(len(Genomes))],Best_parts)
    plt.legend(["Average","Best"])
  plt.xlabel("Generation")
  plt.ylabel("Block num")
  plt.ylim(0,30)
  plt.savefig(os.path.dirname(__file__) + "/partsNum.svg")


  return Most_parts

def fitness(Genomes):
  total_fitness=[]
  for genome in Genomes:
      gen_fitness=[]
      for ind,g in genome:
          gen_fitness.append(g.fitness)

      gen_fitness.sort()
      total_fitness.append(gen_fitness)

  plt.figure()
  plt.plot(np.arange(len(Genomes)),[np.mean(i) for i in total_fitness],np.arange(len(Genomes)),[i[-1] for i in total_fitness])
  plt.ylim(0,65)
  plt.legend(["Average","Best"])
  plt.xlabel("Generation")
  plt.ylabel("Fitness")
