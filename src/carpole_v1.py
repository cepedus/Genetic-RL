import gym
from gnn import GeneticNeuralNetwork

# ====== Setting up Environment ======= #
env = gym.make("CartPole-v1")
env.reset()

# Defining constants
goal_steps = 500  # Our fitness objective
score_requirement = 50  # Our baseline score for random population


# ====== Random agent ====== #

def random_agent():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame (step), up to 200
        for t in range(200):
            # Display environment (optional)
            env.render()

            # Sample action (random)
            action = env.action_space.sample()

            # Execute the action
            observation, reward, done, info = env.step(action)
            if done:
                break


# ====== Creating Population ====== #

def onehot_reshape(game_memory):
    training_data = []
    for data in game_memory:
        # convert to one-hot
        if data[1] == 1:
            output = [0, 1]
        else:
            output = [1, 0]
        training_data.append([data[0], output])
    return training_data


# ====== Genetic Evolution using GNN ====== #

# Define the layers shapes
observation, _, _, _ = env.step(env.action_space.sample())
print(observation.shape)
print(env.action_space.n)  # Only for type Discrete!! gym.spaces.discrete.Discrete
layers_shapes = [observation.shape[0], 4, 2, env.action_space.n]

# Define if the training starts from zero or from pre-trained model
train_from_scratch = False

# Create a List of all active GeneticNeuralNetworks
networks = []
pool = []
# Track Generations
generation = 0
# Initial Population
n = 20

# If we do not have a pre-writen model we start from scratch
if train_from_scratch:
    # Generate n randomly weighted neural networks
    for i in range(n):
        networks.append(GeneticNeuralNetwork(layers_shapes))

    # Cache Max Fitness
    max_fitness = 0

    # Max Fitness Weights
    optimal_weights = []

    # While we haven't crossed our goal we will keep evolving
    while max_fitness < goal_steps:
        # log the current generation
        generation += 1
        print(f'Generation: {generation}')
        max_fitness_generation = 0

        # Forward propagate the neural networks (play cartpole) to compute a fitness score
        for i, nn in enumerate(networks):
            # Propagate to calculate fitness score
            nn.forward_propagation(X_train, y_train)
            if nn.fitness > max_fitness_generation:
                max_fitness_generation = nn.fitness
            # Add to pool after calculating fitness
            pool.append(nn)
        print(f'Max fitness of this generation: {max_fitness_generation}')