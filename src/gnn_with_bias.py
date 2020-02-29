# code inspired from https://medium.com/swlh/genetic-artificial-neural-networks-d6b85578ba99

import random

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm

class GNN_bias(Sequential):

    def __init__(self, layers_shapes, child_weights=None, dropout=0):
        
        super().__init__()
        self.layers_shapes = layers_shapes
        self.fitness = 0
        self.dropout = dropout

        if child_weights is None:
            layers = [Dense(layers_shapes[1], input_shape=(layers_shapes[0],), activation='sigmoid'),
                      Dropout(dropout)]
            for shape in layers_shapes[2:-1]:
                layers.append(Dense(shape, activation='sigmoid'))
                layers.append(Dropout(dropout))
            layers.append(Dense(layers_shapes[-1], activation='softmax'))
            layers.append(Dropout(dropout))
        
        else:
            # Set weights within the layers
            layers = [Dense(layers_shapes[1], input_shape=(layers_shapes[0],), activation='sigmoid',
                            weights=child_weights[0]),
                      Dropout(dropout)]
            for i, shape in enumerate(layers_shapes[2:-1]):
                layers.append(Dense(shape, activation='sigmoid',
                                    weights=child_weights[i + 1]))
                layers.append(Dropout(dropout))
            layers.append(Dense(layers_shapes[-1], activation='softmax',
                                weights=child_weights[-1]))
            layers.append(Dropout(dropout))
        
        for layer in layers:
            self.add(layer)

    def update_weights(self, network_weights):
        count = 0
        for layer in self.layers:
            w = layer.get_weights()
            if w:
                layer.set_weights(network_weights[count])
                count += 1
    
    def run_single(self, env, render=False):
        raise NotImplementedError

    def compile_train(self, X_train, y_train, epochs):
        raise NotImplementedError

def mutation(network_weights, p_mutation=0.7):
    for weights in network_weights:
        if random.uniform(0, 1) < p_mutation:
            try:
                weights *= random.uniform(-5, 5)
            except TypeError:
                pass
    return network_weights
    
def mutate_network(network, p_mutation=0.7):

    nn_weights = []
    for layer in network.layers:
        w = layer.get_weights()
        if w:
            nn_weights.append(w)

    nn_weights = mutation(nn_weights, p_mutation)
    network.update_weights(nn_weights)
    return network

# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(nn1, nn2, p_mutation=0.7):
    # Assert both Neural Networks are of the same format
    assert nn1.layers_shapes == nn2.layers_shapes
    assert nn1.dropout == nn2.dropout

    # Lists for respective weights
    nn1_weights = []
    nn2_weights = []
    child_weights = []
    # Get all weights from all layers in the networks
    for layer_1 in nn1.layers:
        w = layer_1.get_weights()
        if w:
            nn1_weights.append(w)
    for layer_2 in nn2.layers:
        w = layer_2.get_weights()
        if w:
            nn2_weights.append(w)

    for i in range(len(nn1_weights)):
        for j in range(len(nn1_weights[i])):
            nn1_weights[i][0][:, j] = random.choice([nn1_weights[i][0][:, j], nn2_weights[i][0][:, j]])

        # After crossover add weights to child
        child_weights.append(nn1_weights[i])
        child_weights = child_weights
    # add a chance for mutation
    child_weights = mutation(child_weights, p_mutation)

    # Create and return child object
    return GNN_bias(layers_shapes=nn1.layers_shapes, child_weights=child_weights, dropout=nn1.dropout)

def random_pick(population):
    total_fitness = np.sum([gnn.fitness for gnn in population])
    selection_probabilities = [gnn.fitness / total_fitness for gnn in population]
    pick_1 = np.random.choice(len(population), p=selection_probabilities)
    pick_2 = np.random.choice(len(population), p=selection_probabilities)
    return population[pick_1], population[pick_2]

def ranking_pick(population):
    sorted_population = sorted(population, key=lambda gnn: gnn.fitness, reverse=True)
    return sorted_population[:2]

def statistics(population):
    population_fitness = [individual.fitness for individual in population]
    return np.mean(population_fitness), np.min(population_fitness), np.max(population_fitness)