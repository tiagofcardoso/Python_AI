import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

class RouteOptimizer:
    def __init__(self, num_locations=10, pop_size=52, num_generations=100, mutation_rate=0.05):
        # Ajustar pop_size para ser divisível por 4
        self.pop_size = (pop_size // 4) * 4  # Arredonda para o múltiplo de 4 mais próximo
        self.num_locations = num_locations
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        
        # Criar matriz de distâncias aleatória
        self.distance_matrix = np.random.randint(1, 100, size=(num_locations, num_locations))
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Resultados
        self.classic_ga_result = None
        self.nsga_result = None
        self.memetic_result = None
        
    def fitness_function(self, route):
        """Função de fitness para o algoritmo genético clássico"""
        total_distance = sum(self.distance_matrix[route[i], route[i + 1]] 
                           for i in range(len(route) - 1))
        # Adicionar distância de volta ao início
        total_distance += self.distance_matrix[route[-1], route[0]]
        return total_distance
    
    def multi_objective_fitness(self, individual):
        """Função de fitness para NSGA-II (custo e tempo)"""
        distance = self.fitness_function(individual)
        cost = distance * 2  # Custo por km
        time = distance / 50  # Velocidade média de 50 km/h
        return [cost, time]
    
    def run_classic_ga(self):
        """Executa o algoritmo genético clássico"""
        population = [random.sample(range(self.num_locations), self.num_locations) 
                     for _ in range(self.pop_size)]
        best_fitness_history = []
        
        for generation in range(self.num_generations):
            fitness_values = [self.fitness_function(ind) for ind in population]
            selected_indices = np.argsort(fitness_values)[:self.pop_size // 2]
            selected_population = [population[i] for i in selected_indices]
            
            offspring = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = random.sample(selected_population, 2)
                crossover_point = random.randint(1, self.num_locations - 2)
                child = parent1[:crossover_point] + [loc for loc in parent2 
                                                   if loc not in parent1[:crossover_point]]
                offspring.append(child)
            
            for child in offspring:
                if random.random() < self.mutation_rate:
                    idx1, idx2 = random.sample(range(self.num_locations), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
            
            population = selected_population + offspring
            best_fitness = min(fitness_values)
            best_fitness_history.append(best_fitness)
        
        best_idx = np.argmin([self.fitness_function(ind) for ind in population])
        self.classic_ga_result = {
            'route': population[best_idx],
            'fitness': self.fitness_function(population[best_idx]),
            'history': best_fitness_history
        }
        
    def run_nsga2(self):
        """Executa o algoritmo NSGA-II"""
        # Limpar definições anteriores do DEAP
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
            del creator.Individual
            
        # Configurar DEAP
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        toolbox = base.Toolbox()
        
        def create_individual():
            return creator.Individual(random.sample(range(self.num_locations), self.num_locations))
        
        toolbox.register("individual", create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.multi_objective_fitness)
        toolbox.register("mate", tools.cxOrdered)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        # População inicial
        population = toolbox.population(n=self.pop_size)
        
        # Avaliar população inicial
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Evolução
        for gen in range(self.num_generations):
            # Criar offspring usando crossover e mutação
            offspring = []
            for _ in range(self.pop_size):
                if len(offspring) >= self.pop_size:
                    break
                # Selecionar pais
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                
                # Crossover
                child1, child2 = toolbox.mate(list(parent1), list(parent2))
                child1 = creator.Individual(child1)
                
                # Mutação
                if random.random() < self.mutation_rate:
                    toolbox.mutate(child1)
                
                offspring.append(child1)
            
            # Avaliar offspring
            for ind in offspring:
                ind.fitness.values = toolbox.evaluate(ind)
            
            # Seleção para próxima geração
            population = toolbox.select(population + offspring, self.pop_size)
        
        self.nsga_result = {
            'population': population,
            'pareto_front': [ind.fitness.values for ind in population]
        }
    
    def run_memetic(self):
        """Executa o algoritmo memético"""
        population = [random.sample(range(self.num_locations), self.num_locations) 
                     for _ in range(self.pop_size)]
        best_fitness_history = []
        
        for generation in range(self.num_generations):
            # GA steps
            fitness_values = [self.fitness_function(ind) for ind in population]
            selected_indices = np.argsort(fitness_values)[:self.pop_size // 2]
            selected_population = [population[i] for i in selected_indices]
            
            offspring = []
            for _ in range(self.pop_size // 2):
                parent1, parent2 = random.sample(selected_population, 2)
                crossover_point = random.randint(1, self.num_locations - 2)
                child = parent1[:crossover_point] + [loc for loc in parent2 
                                                   if loc not in parent1[:crossover_point]]
                if random.random() < self.mutation_rate:
                    idx1, idx2 = random.sample(range(self.num_locations), 2)
                    child[idx1], child[idx2] = child[idx2], child[idx1]
                offspring.append(child)
            
            # Local search
            improved_population = []
            for ind in selected_population + offspring:
                # 2-opt local search
                improved = True
                while improved:
                    improved = False
                    for i in range(1, len(ind) - 2):
                        for j in range(i + 1, len(ind)):
                            new_route = ind[:]
                            new_route[i:j] = ind[j-1:i-1:-1]
                            if self.fitness_function(new_route) < self.fitness_function(ind):
                                ind = new_route
                                improved = True
                improved_population.append(ind)
            
            population = improved_population
            best_fitness = min(fitness_values)
            best_fitness_history.append(best_fitness)
        
        best_idx = np.argmin([self.fitness_function(ind) for ind in population])
        self.memetic_result = {
            'route': population[best_idx],
            'fitness': self.fitness_function(population[best_idx]),
            'history': best_fitness_history
        }
    
    def visualize_results(self):
        """Visualiza os resultados dos três algoritmos"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Convergência do GA Clássico
        plt.subplot(131)
        plt.plot(self.classic_ga_result['history'])
        plt.title('GA Clássico: Convergência')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        
        # Plot 2: Frente de Pareto do NSGA-II
        plt.subplot(132)
        pareto_front = np.array(self.nsga_result['pareto_front'])
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
        plt.title('NSGA-II: Frente de Pareto')
        plt.xlabel('Custo')
        plt.ylabel('Tempo')
        
        # Plot 3: Comparação Memético vs Clássico
        plt.subplot(133)
        plt.plot(self.classic_ga_result['history'], label='GA Clássico')
        plt.plot(self.memetic_result['history'], label='Memético')
        plt.title('Comparação de Convergência')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Criar e executar o otimizador
    optimizer = RouteOptimizer(num_locations=10, pop_size=52, num_generations=100)
    
    print("Executar GA Clássico...")
    optimizer.run_classic_ga()
    
    print("Executar NSGA-II...")
    optimizer.run_nsga2()
    
    print("Executar Algoritmo Memético...")
    optimizer.run_memetic()
    
    # Mostrar resultados
    print("\nResultados:")
    print(f"GA Clássico - Melhor fitness: {optimizer.classic_ga_result['fitness']}")
    print(f"Memético - Melhor fitness: {optimizer.memetic_result['fitness']}")
    print(f"NSGA-II - Número de soluções não-dominadas: {len(optimizer.nsga_result['pareto_front'])}")
    
    # Visualizar resultados
    optimizer.visualize_results()