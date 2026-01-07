import time

def reporter(fitness_function, genomes, mode, camera = None):

  start = time.time()

  fitness_function(genomes,mode, camera = camera)

  elapsed = time.time() - start

  fitness = [g.fitness for _,g in genomes]

  print()
  print("Excution Size : "+str(len(genomes))+" Best : "+ str(round(max(fitness), 3))+" Average: "+str(round(sum(fitness)/len(fitness), 3)))
  print(" time: {0:.3f} sec ".format(elapsed))

