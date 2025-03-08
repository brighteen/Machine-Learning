import random

from selection_rank_with_elite import selection_rank_with_elite
from individual import Individual

POPULATION_SIZE = 5
random.seed(3)

population = Individual.create_random_population(POPULATION_SIZE)
selected = selection_rank_with_elite(population, elite_size = 2)

print(f"Population: {population}")
print(f"Population: {selected}")