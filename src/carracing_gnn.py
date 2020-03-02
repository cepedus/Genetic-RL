import gym
import copy
from gnn import GeneticNeuralNetwork, random_pick, ranking_pick, dynamic_crossover, mutate_network
from population_gnn import Population
import numpy as np
from time import time

import os


class CarRacingGNN(GeneticNeuralNetwork):

    def run_single(self, env, render=False):
        obs = env.reset()
        fitness = 0
        while True:
            if render:
                env.render()
            action = self.predict(np.array([np.array(obs).reshape(-1, )]))[0]
            if np.isnan(action).any():
                break
            else:
                obs, reward, done, _ = env.step(action)
                fitness += reward
                if done:
                    break
        self.fitness = fitness
        return fitness


def run_generation(env, old_population, new_population, p_mutation):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1, parent2 = random_pick(old_population)

        # Crossover and Mutation
        child1 = dynamic_crossover(parent1, parent2, p_mutation)
        child2 = dynamic_crossover(parent1, parent2, p_mutation)

        # Inherit casting TODO: Bad practice... Do it properly
        child1.__class__ = CarRacingGNN
        child2.__class__ = CarRacingGNN

        # Run childs
        child1.run_single(env)
        child2.run_single(env)

        # If children fitness is greater than parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env.seed(123)
    np.random.seed(int(time() * 1e9) % 4294967296)
    env._max_episode_steps = 700
    discrete = (type(env.action_space) == gym.spaces.discrete.Discrete)

    POPULATION_SIZE = 4
    MAX_GENERATION = 2
    MUTATION_RATE = 0.8
    obs = env.reset().reshape(-1,)
    output_shape = env.action_space.n if discrete else env.action_space.shape
    layers_shapes = [obs.shape[0], 6, output_shape[0]]
    dropout_rate = 0.1
    baseline_fitness = 0

    initial_network = CarRacingGNN(layers_shapes, dropout=dropout_rate, discrete=discrete)

    # Mutate network until minimum performance
    t0 = time()
    initial_fitness = initial_network.run_single(env)
    while initial_fitness < baseline_fitness:
        initial_network = mutate_network(initial_network, 0.8)
        initial_fitness = initial_network.run_single(env)
        print(initial_fitness)
    print('Ancestral Fitness: ', initial_fitness, ' found in ', time()-t0, 'ms')

    p = Population(initial_network,
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)

    # Folder name for good ol' Windows
    dirname = os.path.dirname(__file__)
    out_folder = filename = os.path.join(dirname, '../models/carracing/')

    p.run(env, run_generation, verbose=True, output_folder=out_folder, log=True, render=True)

    env.close()
