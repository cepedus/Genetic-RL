from typing import Callable
from copy import deepcopy, copy
from gnn import statistics
from datetime import datetime
from getpass import getuser
from gnn import mutate_network


class Population:
    def __init__(self, gnn, pop_size, max_generation, p_mutation):
        self.pop_size = pop_size  # Population size (number of neural networks in a generation/population)
        self.max_generation = max_generation  # Max number of generations until the end
        self.p_mutation = p_mutation  # Proba of mutation
        self.old_population = [mutate_network(gnn) for _ in range(pop_size)]
        self.new_population = []

#    @timing
    def run(self, env, run_generation: Callable, verbose=False, log=False, output_folder=None):
        for i in range(self.max_generation):
            fitnesses = [p.run_single(env) for p in self.old_population]
            best_from_old_population = self.old_population[fitnesses.index(max(fitnesses))]
            best_from_old_population.run_single(env, render=True)
            self.new_population = [None for _ in range(self.pop_size)]

            run_generation(env,
                           self.old_population,
                           self.new_population,
                           self.p_mutation)

            if log:
                self.save_logs(i, output_folder)
            if verbose:
                self.show_stats(i)

            self.old_population = copy(self.new_population)

        # TODO: save model  # import pickle / joblib

    def save_logs(self, n_gen, output_folder):
        """
        CSV format -> date,n_generation,mean,min,max
        """
        date = datetime.now().strftime('%m-%d-%Y_%H-%M')
        file_name = 'logs.csv'
        username = getuser()
        mean, min, max = statistics(self.new_population)
        stats = f'{date},{n_gen},{mean},{min},{max}\n'
        with open(output_folder + username + '-' + file_name, 'a') as f:
            f.write(stats)

    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = datetime.now().strftime('%m-%d-%Y_%H-%M')
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print(stats)

    def show_best(self):
        return