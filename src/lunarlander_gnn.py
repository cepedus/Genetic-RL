from abc import ABC

import gym
import copy
from gnn import GeneticNeuralNetwork, random_pick, ranking_pick, dynamic_crossover, mutate_network, run_generation
from population_gnn import Population
import numpy as np
from time import time

import os


class LunarLanderGNN(GeneticNeuralNetwork):
    pass


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(123)
    np.random.seed(int(time() * 1e9) % 4294967296)
    env._max_episode_steps = 700

    POPULATION_SIZE = 6  # must be even
    MAX_GENERATION = 20
    MUTATION_RATE = 0.7
    obs = env.reset()
    layers_shapes = [obs.shape[0], 16, env.action_space.n]
    dropout_rate = 0.1
    baseline_fitness = -50

    initial_network = LunarLanderGNN(layers_shapes, dropout=dropout_rate)
    print('created GNN, looking for ancestral fitness')
    # Mutate network until minimum performance
    t0 = time()
    initial_fitness = initial_network.run_single(env)
    while initial_fitness < baseline_fitness:
        initial_network = mutate_network(initial_network, MUTATION_RATE)
        initial_fitness = initial_network.run_single(env)
        print(initial_fitness)
    print('Ancestral Fitness: ', initial_fitness, ' found in ', time()-t0, 'ms')
    initial_network.run_single(env, render=True)
    p = Population(initial_network,
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)
    # Folder name for good ol' Windows
    dirname = os.path.dirname(__file__)
    out_folder = filename = os.path.join(dirname, '../models/lunarlander/')
    
    p.run(env, run_generation, random_selection=False, verbose=True, output_folder=out_folder, log=True, render=True)

    env.close()
