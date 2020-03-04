import gym
from gnn import GeneticNeuralNetwork, mutate_network, run_generation, baseline_init
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

    print('creating GNN fulfilling minimum baseline fitness')
    # Mutate network until minimum performance
    initial_network = baseline_init(LunarLanderGNN(layers_shapes, dropout=dropout_rate),
                                    env, baseline_fitness, render=True)

    p = Population(initial_network,
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)
    # Folder name for good ol' Windows
    dirname = os.path.dirname(__file__)
    out_folder = filename = os.path.join(dirname, '../models/lunarlander/')
    
    p.run(env, run_generation, random_selection=False, verbose=True, output_folder=out_folder, log=True, render=True)

    env.close()
