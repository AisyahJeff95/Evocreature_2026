import os
import pickle
from evaluation import eval_fitness
import matplotlib.pyplot as plt
import numpy as np
from pureples.shared.visualize import draw_net_3d
import glob
import creature.plot2 as plot
import pybullet as p


gen=1999
local_dir = os.path.dirname(__file__)

genomes_file_path=glob.glob(local_dir+"/genomes.pkl")[0]
with open(genomes_file_path, "rb") as f:
    Genomes = pickle.load(f)
genomes = Genomes[gen]


# Find the best individual across all generations
best_overall = max((g.fitness, ind, gen) for gen, genomes in enumerate(Genomes) for ind, g in genomes)
# Unpack the tuple
best_fitness, best_ind, best_gen = best_overall
# Load the best generation for further analysis if needed
best_generation = Genomes[best_gen]
# Evaluate fitness if needed
eval_fitness(best_generation, mode="DIRECT")


# print(f"num of Nicheboard ={len(Nicheboard)}")

# print(f"num of niche_board ={len(niche_board)}")
# print(f"niche_board ={niche_board}")


plot.trajectory(60, best_generation)
# plot.nicheBoard(60, niche_board)
# plot.nicheBoardAll(60, Nicheboard)
plot.event(60, Event)
plot.partsNum(Genomes)
plot.fitness(Genomes)
plt.show()