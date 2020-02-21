# code inspired from https://medium.com/swlh/genetic-artificial-neural-networks-d6b85578ba99

# ======= Defining our Genetic Neural Network ======= #
import random

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm


class GeneticNeuralNetwork(Sequential):
    # Constructor
    def __init__(self, layers_shapes, child_weights=None):
        # Initialize Sequential Model Super Class
        super().__init__()

        # Add layers_shape as a property of our Neural Network
        self.layers_shapes = layers_shapes
        self.fitness = 0

        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layers = [Dense(layers_shapes[1], input_shape=(layers_shapes[0],), activation='sigmoid')]
            for shape in layers_shapes[2:-1]:
                layers.append(Dense(shape, activation='sigmoid'))
            layers.append(Dense(layers_shapes[-1], activation='sigmoid'))

        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            layers = [Dense(layers_shapes[1], input_shape=(layers_shapes[0],), activation='sigmoid',
                            weights=[child_weights[0], np.zeros(layers_shapes[1])])]
            for i, shape in enumerate(layers_shapes[2:-1]):
                layers.append(Dense(shape, activation='sigmoid',
                                    weights=[child_weights[i+1], np.zeros(shape)]))
            layers.append(Dense(layers_shapes[-1], activation='sigmoid',
                                weights=[child_weights[-1], np.zeros(layers_shapes[-1])]))
        for layer in layers:
            self.add(layer)

    # Function for foward propagating a row vector of a matrix
    def forward_propagation(self, X_train, y_train):
        # Forward propagation
        y_pred = self.predict(X_train).reshape(-1,).round()
        # Compute fitness score
        self.fitness = accuracy_score(y_train, y_pred)

    # Standard Backpropagation
    def compile_train(self, X_train, y_train, epochs):
        self.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        self.fit(X_train, y_train, epochs=epochs)

    @staticmethod
    def mutation(weights):
        # weights is a NxM matrix
        for i, line in enumerate(weights):
            for j in range(len(line)):
                if random.uniform(0, 1) > 0.9:
                    weights[i][j] *= random.uniform(2, 5)
        return weights
    # Crossover traits between two Genetic Neural Networks
    def dynamic_crossover(self, nn2):

        # Assert both Neural Networks are of the same format
        assert self.layers_shapes == nn2.layers_shapes

        # Lists for respective weights
        nn1_weights = []
        nn2_weights = []
        child_weights = []
        # Get all weights from all layers in the networks
        for layer_1 in self.layers:
            nn1_weights.append(layer_1.get_weights()[0])
        for layer_2 in nn2.layers:
            nn2_weights.append(layer_2.get_weights()[0])

        for i in range(0, len(nn1_weights)):
            for j in range(np.shape(nn1_weights[i])[1] - 1):
                nn1_weights[i][:, j] = random.choice([nn1_weights[i][:, j], nn2_weights[i][:, j]])

            # After crossover add weights to child
            child_weights.append(nn1_weights[i])

        # add a chance for mutation
        child_weights = GeneticNeuralNetwork.mutation(child_weights)

        # Create and return child object
        return GeneticNeuralNetwork(layers_shapes=self.layers_shapes, child_weights=child_weights)


if __name__ == '__main__':
    # Load Data
    data = pd.read_csv('banknote.csv')

    # Create Matrix of Independent variables
    X = np.array(data.drop(['Y'], axis=1))
    # Create vector of dependent variable
    y = np.array(data['Y'])

    # Create a Train Test Split for Genetic Optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Define the layers shapes
    layers_shapes = [X.shape[1], 4, 2, 1]

    # Define if the training starts from zero or from pre-trained model
    train_from_scratch = False

    # Create a List of all active GeneticNeuralNetworks
    networks = []
    pool = []
    # Track Generations
    generation = 0
    # Initial Population
    n = 20

    if not train_from_scratch:
        gnn = GeneticNeuralNetwork(layers_shapes)
        gnn.load_weights('trained_model.h5')

        optimal_weights = []
        for layer in gnn.layers:
            optimal_weights.append(layer.get_weights()[0])

        max_fitness = gnn.fitness

        # Create n-1 mutations of our model and add to networks
        for i in range(n):
            networks.append(gnn.dynamic_crossover(gnn))

    else:

        # Generate n randomly weighted neural networks
        for i in range(n):
            networks.append(GeneticNeuralNetwork(layers_shapes))

        # Cache Max Fitness
        max_fitness = 0

        # Max Fitness Weights
        optimal_weights = []

    # Evolution Loop
    while max_fitness < .99:
        # log the current generation
        generation += 1
        print(f'Generation: {generation}')
        max_fitness_generation = 0
        # Forward propagate the neural networks to compute a fitness score
        for i, nn in enumerate(networks):
            # Propagate to calculate fitness score
            nn.forward_propagation(X_train, y_train)
            if nn.fitness > max_fitness_generation:
                max_fitness_generation = nn.fitness
            # Add to pool after calculating fitness
            pool.append(nn)
        print(f'Max fitness of this generation: {max_fitness_generation}')
        # Clear for propagation of next children
        networks.clear()

        # Sort based on fitness
        pool = sorted(pool, key=lambda x: x.fitness)
        pool.reverse()

        # Find Max Fitness and log associated weights
        for i in range(len(pool)):
            # If there is a new max fitness among the population
            if pool[i].fitness > max_fitness:
                max_fitness = pool[i].fitness
                print(f'Max Fitness: {max_fitness}')
                # Reset optimal_weights
                # Iterate through layers and append the layers' weights to optimal
                for layer in pool[i].layers:
                    optimal_weights.append(layer.get_weights()[0])
                print(optimal_weights)

        # Crossover, top 4 randomly select 2 partners for child
        print('Mixing the top 4 individuals...')
        for i in tqdm(range(4)):
            for j in range(2):
                # Create a child and add to networks
                temp = pool[i].dynamic_crossover(random.choice(pool))
                # Add to networks to calculate fitness score next iteration
                networks.append(temp)

    # Create a Genetic Neural Network with optimal inital weights
    gnn = GeneticNeuralNetwork(layers_shapes, optimal_weights)
    gnn.compile_train(X_train, y_train, 10)

    # Serialize weights to HDF5
    gnn.save_weights('trained_model.h5')
    # Test the Genetic Neural Network Out of Sample
    y_hat = gnn.predict(X_test).reshape(-1,).round()
    print(f'Test Accuracy: {accuracy_score(y_test, y_hat)}')
