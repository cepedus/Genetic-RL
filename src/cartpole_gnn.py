import gym
from gnn import GeneticNeuralNetwork, mutate_network, run_generation, baseline_init
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

    # If you want to load a pre-existing model:
    # initial_network = CartPoleGNN.load_model(out_folder + '03-02-2020_14-43')
    # Mutate network until minimum performance
    t0 = time()
    initial_network = baseline_init(CartPoleGNN(layers_shapes, dropout=dropout_rate),
                                    env, baseline_fitness, render=False)

    p = Population(initial_network,                     # Define our population
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)

    p.run(env, run_generation, random_selection=False, verbose=False, output_folder=out_folder, log=True, render=False)  # Run evolution

    env.close()

    print('done in', time()-t0)

