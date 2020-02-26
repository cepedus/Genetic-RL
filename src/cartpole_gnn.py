import gym
import copy
from gnn import GeneticNeuralNetwork, random_pick, ranking_pick, dynamic_crossover
from population_gnn import Population
import numpy as np


class CartPoleGNN(GeneticNeuralNetwork):

    def run_single(self, env, n_episodes=600, render=False):
        obs = env.reset()
        fitness = 0
        for _ in range(n_episodes):
            if render:
                env.render()
            action_dist = self.predict(np.array([np.array(obs).reshape(-1, )]))[0]
            # action = action_dist.index(np.random.choice(action_dist, p=action_dist))
            action = np.argmax(action_dist)  # ############################ TODO: take random action from distribution
            obs, reward, done, _ = env.step(round(action.item()))
            fitness += reward
            if done:
                break
        self.fitness = fitness
        return fitness


def run_generation(env, old_population, new_population, p_mutation):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        parent1, parent2 = ranking_pick(old_population)

        # Crossover and Mutation
        child1 = dynamic_crossover(parent1, parent2, p_mutation)
        child2 = dynamic_crossover(parent1, parent2, p_mutation)

        # Inherit casting TODO: Bad practice... Do it properly
        child1.__class__ = CartPoleGNN
        child2.__class__ = CartPoleGNN

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
    env = gym.make('CartPole-v1')
    env.seed(123)

    POPULATION_SIZE = 10
    MAX_GENERATION = 20
    MUTATION_RATE = 0.7
    obs = env.reset()
    layers_shapes = [obs.shape[0], 4, env.action_space.n]
    dropout_rate = 0.1

    p = Population(CartPoleGNN(layers_shapes, dropout=dropout_rate),
                   POPULATION_SIZE,
                   MAX_GENERATION,
                   MUTATION_RATE)
    p.run(env, run_generation, verbose=True, output_folder='../models/cartpole', log=True)

    env.close()
