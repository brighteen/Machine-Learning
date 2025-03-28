import random
from selection_tournament import selection_tournament
from individual import Individual

POPULATION_SIZE = 5
random.seed(4)

population = Individual.create_random_population(POPULATION_SIZE)
selected = selection_tournament(population, group_size = 3)

print(f"Population: {population}")
print(f"Selected: {selected}")
