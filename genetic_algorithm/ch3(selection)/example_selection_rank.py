import random

from selection_rank import selection_rank
from individual import Individual

POPULATION_SIZE = 5
random.seed(18)

population = Individual.create_random_population(POPULATION_SIZE)
selected = selection_rank(population)

print(f'Population: {population}')
print(f'Selected: {selected}')
