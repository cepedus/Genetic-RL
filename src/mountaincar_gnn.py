import gym
from gnn import GeneticNeuralNetwork, mutate_network, run_generation
from population_gnn import Population
import numpy as np
from time import time

import os


class MountainCarGNN(GeneticNeuralNetwork):

    def run_single(self, env, render=False):
        obs = env.reset()
        fitness = 0
        while True:
            if render:
                env.render()
            action_dist = self.predict(np.array([np.array(obs).reshape(-1, )]))[0]
            if np.isnan(action_dist).any():
                break
            else:
                action = np.random.choice(np.arange(env.action_space.n), p=action_dist)
                # action = np.argmax(action_dist)  # TODO: Test if taking a random action is better than the most prob
                obs, reward, done, _ = env.step(round(action.item()))
                
                if obs[0] > -0.2:
                    reward = 1
                if obs[0] > 0.0:
                    reward = 2
                if obs[0] > 0.2:
                    reward = 3

                fitness += reward
                if done:
                    break
        self.fitness = fitness
        return fitness


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(123)
    np.random.seed(int(time() * 1e9) % 4294967296)
    env._max_episode_steps = 700

    POPULATION_SIZE = 6
    MAX_GENERATION = 20
    MUTATION_RATE = 0.8
    obs = env.reset()
    layers_shapes = [obs.shape[0], 10, env.action_space.n]
    dropout_rate = 0.1
    baseline_fitness = -100

    initial_network = MountainCarGNN(layers_shapes, dropout=dropout_rate)

    # # # Mutate network until minimum performance
    print('created GNN, looking for ancestral fitness')
    t0 = time()
    initial_fitness = initial_network.run_single(env)
    while initial_fitness < baseline_fitness:
        initial_network = mutate_network(initial_network, 0.8)
        initial_fitness = initial_network.run_single(env)
        print(initial_fitness)
    print('Ancestral Fitness: ', initial_fitness, ' found in ', time()-t0, 's')
    initial_network.run_single(env, render=False)

    p = Population(initial_network,
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)

    print('created population')
    # Folder name for good ol' Windows
    dirname = os.path.dirname(__file__)
    out_folder = filename = os.path.join(dirname, '../models/mountaincar/')

    print('running')
    
    p.run(env, run_generation, random_selection=False, verbose=True, output_folder=out_folder, log=True, render=False)

    env.close()
