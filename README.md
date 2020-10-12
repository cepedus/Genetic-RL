# Genetic Neural Network on OpenAI Gym environments

- Luiz Bezerra
- Martín Cepeda
- Raphael Mendes

Reinforcement learning is one of the paradigms of machine learning these days besides supervised and unsupervised learning, where we have an virtual agent in a given environment that learns how to act based on rewards it gets. In this project, we test the performance of the Genetic Neural Network and variants over some already coded environments from OpenAI Gym to give insights about Evolutionary Strategies in RL environments.

Environments choisis:
http://gym.openai.com/envs/CartPole-v1
http://gym.openai.com/envs/MountainCarContinuous-v0
http://gym.openai.com/envs/BipedalWalker-v2
http://gym.openai.com/envs/BipedalWalkerHardcore-v2
http://gym.openai.com/envs/LunarLanderContinuous-v2

The basic architecture of our Agent is found at the file `gnn.py`, it is the class `GeneticNeuralNetwork`. The creation of a new agent can be inspired from the file `cartpole_gnn.py` in which we declare a new class (agent) called `CartPoleGNN`. The implementation of the function `run_single` is optional (if one wants to change the reward for example), this function simulates the behavior of this agent in the environment. To instantiate an agent you must define two variables: 

- `layers_shapes`: a list containing the shapes of the layers in the neural network, ex: layers shapes = `[4,6,2]` for a Network that has 4 inputs, 6 hidden nodes and 2 possible actions. For this, we take shapes `[state dimension,h0, ... ,hk,action dimension]` where hi the number of hidden neurons can vary both in depth (number of hidden layers k) and quantity. 
- `dropout_rate`: a float in `[0,1]` indicating the dropout rate after each layer (except for the last one) for regularization purposes. The implementation alongside with the Evolutionary Strategy can be find in the main function of `cartpole_gnn.py` that is well commented.

We observed that our implementation of Genetic Neural Networks can solve simple tasks as in the first two environments `CartPole` and `MountainCar`. For more complex problems, we did not get a good result as observed in the environment `LunarLander`. Another important observation about our algorithm is that it is not stable as we can see at the graphics of the score. However, we already tried new approaches for mutation and crossover that seems to work better, but we still do not have mathematical and experimental background to prove that it is more stable.
