Particle Swarm Optimization
Code:-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Rastrigin function
def rastrigin(x):
    n = len(x)
    return 10*n + sum([xi**2 - 10*np.cos(2*np.pi*xi) for xi in x])

# Define the PSO algorithm
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness

# Define the dimensions of the problem
dim = 2

# Run the PSO algorithm on the Rastrigin function
solution, fitness = pso(rastrigin, dim=dim)

# Print the solution and fitness value
print('Solution:', solution)
print('Fitness:', fitness)

# Create a meshgrid for visualization
x = np.linspace(-5.12, 5.12, 100)
y = np.linspace(-5.12, 5.12, 100)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

# Create a 3D plot of the Rastrigin function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot the solution found by the PSO algorithm
ax.scatter(solution[0], solution[1], fitness, color='red')
plt.show()


Genetic Algorithm
Code:-
import random

# Define the fitness function
def fitness_function(chromosome):
    return sum(chromosome)

# Define the genetic algorithm
def genetic_algorithm(population_size, chromosome_length, fitness_function, mutation_probability=0.1):
    # Initialize the population with random chromosomes
    population = [[random.randint(0, 1) for j in range(chromosome_length)] for i in range(population_size)]
    
    # Loop until a satisfactory solution is found
    while True:
        # Evaluate the fitness of each chromosome in the population
        fitness_values = [fitness_function(chromosome) for chromosome in population]
        
        # Select the fittest chromosomes to be the parents of the next generation
        parents = [population[i] for i in range(population_size) if fitness_values[i] == max(fitness_values)]
        
        # Generate the next generation by applying crossover and mutation to the parents
        next_generation = []
        while len(next_generation) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            crossover_point = random.randint(1, chromosome_length - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            for i in range(chromosome_length):
                if random.random() < mutation_probability:
                    child[i] = 1 - child[i]
            next_generation.append(child)
        
        # Update the population with the next generation
        population = next_generation
        
        # Check if a satisfactory solution has been found
        if max(fitness_values) == chromosome_length:
            return parents[0]

# Example usage
chromosome_length = 10
population_size = 50
solution = genetic_algorithm(population_size, chromosome_length, fitness_function)
print("Solution found:", solution)
