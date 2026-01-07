import neat
import neat.nn
import pickle
from pureples.shared.visualize import draw_net_3d
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.update_config import update_config
from pureples.hyperneat import create_phenotype_network
import copy
import matplotlib.pyplot as plt

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

input_coordinates  = [(-1.0, -1.0, -1),(0.0, -1.0, 0),(1.0, -1.0, 1)]
hidden_coordinates = [[(1,1,1)]]#[[(-1.0, 0.0, 0), (0.0, 0.0, 0), (1.0, 0.0, 2)]]
output_coordinates = [(0.0, 1.0, 1)]
activations = len(hidden_coordinates) + 2

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

file_name="config_cppn_xor"

#update_config(file_name, sub)

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            file_name)

def eval_fitness(genomes, config):

    for idx, g in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(g, config)
        g.substrate=copy.deepcopy(sub)

        f=0
        while f<10:
            ## もしノードが追加されるならsubを更新してこれを繰り返す
            g.substrate.input_coordinates.append((f,1,1))
            g.substrate.output_coordinates.append((1,1,f))
            net = create_phenotype_network(cppn, g.substrate)
            
            sum_square_error = 0.0
            for inputs, expected in zip(xor_inputs, xor_outputs):

                new_input = inputs + (1.0,)
                for i in range(f+1):
                    new_input += (f,)
                net.reset()
                for i in range(activations):
                    
                    ##形態形成にはcppn = neat.nn.FeedForwardNetwork.create(g, config)を使う
                    ## これはHyperNEATの出力
                    output = net.activate(new_input)

                sum_square_error += ((output[0] - expected[0])**2.0)/4.0
            f += 1
 
        g.fitness = 1 - sum_square_error


# If run as script.
def run(gens):
    pop = neat.population.Population(config)
    # species = neat.species(pop)
    stats = neat.statistics.StatisticsReporter()
    # stats.get_species_fitness(null_value='')
    # stats.get_species_sizes()
    stats.save_species_fitness(delimiter=' ', null_value='NA', filename='species_fitness.csv')
    stats.save_species_count(delimiter=' ', filename='speciation.csv')

    # stats.post_evaluate(config, pop, species, best_genome)
    
    # stats.save()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True)) #to print out reports during run

    # print("hyperneat_xor done")
    # print(f"speciesfitness: {species_fitness}")
    # print(f"speciescounts: {species_counts}")
    # stats.save_species_count(' ','/Users/sitiaisyahjaafar/Desktop/niche_evo_single_species/speciation.csv')

    winner = pop.run(eval_fitness, gens)
    print("hyperneat_xor done")
    return winner, stats

if __name__ == '__main__':
    winner = run(10)[0] #run 50 generations
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_net = create_phenotype_network(cppn, winner.substrate)
    f=0
    while f<10:
        for inputs, expected in zip(xor_inputs, xor_outputs):
            new_input = inputs + (1.0,)
            for i in range(f+1):
                new_input += (f,)
            winner_net.reset()
        f+=1
    for i in range(activations):
        output = winner_net.activate(new_input)
    print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))
    
    # Save CPPN if wished reused and draw it to file along with the winner.
    # with open('hyperneat_xor_cppn.pkl', 'wb') as output:
    #     pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
    # draw_net(cppn, filename="hyperneat_xor_cppn")
    # draw_net_3d(winner_net, winner.substrate ,filename="hyperneat_xor_winner")
