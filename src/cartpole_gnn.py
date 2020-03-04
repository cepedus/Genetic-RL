import gym
from gnn import GeneticNeuralNetwork, mutate_network, run_generation
from population_gnn import Population
import numpy as np
from time import time

import os


class CartPoleGNN(GeneticNeuralNetwork):
    pass


if __name__ == '__main__':
    # Definition of the Gym Environment
    env = gym.make('CartPole-v1')
    env.seed(123)
    np.random.seed(int(time() * 1e9) % 4294967296)
    env._max_episode_steps = 700

    POPULATION_SIZE = 6   # Number of individuals per generation
    MAX_GENERATION = 10   # Max number of generations
    MUTATION_RATE = 0.6   # Mutation chance
    obs = env.reset()
    layers_shapes = [obs.shape[0], 4, env.action_space.n]  # Format of the neural network
    dropout_rate = 0.1   # Chance of dropout
    baseline_fitness = 200   # Minimum baseline for the optimized random initialization

    # Folder name for good ol' Windows
    dirname = os.path.dirname(__file__)
    out_folder = filename = os.path.join(dirname, '../models/cartpole/')

    initial_network = CartPoleGNN(layers_shapes, dropout=dropout_rate)   # Instantiate
    # If you want to load a pre-existing model:
    # initial_network = CartPoleGNN.load_model(out_folder + '03-02-2020_14-43')

    # Mutate network until minimum performance
    t0 = time()
    initial_fitness = initial_network.run_single(env)    # Get the initial fitness
    while initial_fitness < baseline_fitness:            # Mutate network until one individual achieves the baseline
        initial_network = mutate_network(initial_network, 0.8)
        initial_fitness = initial_network.run_single(env)
    print('Ancestral Fitness: ', initial_fitness, ' found in ', time()-t0, 'ms')
    initial_network.run_single(env, render=True)        # Show simulation for this optimized initialization

    p = Population(initial_network,                     # Define our population
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)

    p.run(env, run_generation, random_selection=False, verbose=True, output_folder=out_folder, log=True, render=True)  # Run evolution

    env.close()
