import csv
from typing import (
    Any,
    Callable,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
)
import time
from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.api import Representation
from geneticengine.core.representations.tree.treebased import (
    treebased_representation,
    relabel_nodes_of_trees,
)
from geneticengine.algorithms.gp.individual import Individual
import geneticengine.algorithms.gp.generation_steps.selection as selection
import geneticengine.algorithms.gp.generation_steps.mutation as mutation
import geneticengine.algorithms.gp.generation_steps.cross_over as cross_over
from geneticengine.algorithms.gp.callback import Callback


class GP(object):
    # reason for union with noreturn in evaluation function, elitism and elitism: https://stackoverflow.com/questions/51811024/mypy-type-checking-on-callable-thinks-that-member-variable-is-a-method
    grammar: Grammar
    representation: Representation[Any]
    evaluation_function: Union[NoReturn, Callable[[Any], float]]
    random: RandomSource
    population_size: int
    elitism: Union[
        NoReturn,
        Callable[[List[Individual], Callable[[Individual], float]], List[Individual]],
    ]
    mutation: Union[
        NoReturn,
        Callable[[Individual], Individual],
    ]
    max_depth: int
    novelty: Union[NoReturn, Callable[[int], List[Individual]]]
    minimize: bool
    final_population: List[Individual]
    callbacks: List[Callback]

    def __init__(
        self,
        grammar: Grammar,
        evaluation_function: Callable[[Any], float],
        representation: Representation = treebased_representation,
        randomSource: Callable[[int], RandomSource] = RandomSource,
        population_size: int = 200,
        n_elites: int = 5,  # Shouldn't this be a percentage of population size?
        n_novelties: int = 10,
        number_of_generations: int = 100,
        max_depth: int = 15,
        favor_less_deep_trees: bool = False,  # now based on depth, maybe on number of nodes?
        selection_method: Tuple[str, int] = ("tournament", 5),
        # -----
        # As given in A Field Guide to GP, p.17, by Poli and Mcphee
        probability_mutation: float = 0.01,
        probability_crossover: float = 0.9,
        # -----
        hill_climbing: bool = False,
        minimize: bool = False,
        target_fitness: Optional[float] = None,
        force_individual: Any = None,
        seed: int = 123,
        # -----
        timer_stop_criteria: bool = False,  # TODO: This should later be generic
        safe_gen_to_csv: Tuple[str, str] = ("", "all"),
        callbacks: List[Callback] = [],
    ):
        # Add check to input numbers (n_elitism, n_novelties, population_size)
        self.grammar = grammar
        self.representation = representation
        self.evaluation_function = evaluation_function
        self.random = randomSource(seed)
        self.seed = seed
        self.population_size = population_size
        self.elitism = selection.create_elitism(n_elites)
        self.max_depth = max_depth
        self.favor_less_deep_trees = favor_less_deep_trees
        self.novelty = selection.create_novelties(
            self.create_individual, max_depth=max_depth
        )
        self.minimize = minimize
        self.target_fitness = target_fitness
        self.timer_stop_criteria = timer_stop_criteria
        self.safe_gen_to_csv = safe_gen_to_csv
        self.callbacks = callbacks
        if hill_climbing:
            self.mutation = mutation.create_hill_climbing_mutation(
                self.random,
                self.representation,
                self.grammar,
                max_depth,
                self.keyfitness(),
                5,
            )
        else:
            self.mutation = mutation.create_mutation(
                self.random, self.representation, self.grammar, max_depth
            )
        self.cross_over = cross_over.create_cross_over(
            self.random, self.representation, self.grammar, max_depth
        )
        self.n_novelties = n_novelties
        self.number_of_generations = number_of_generations
        self.probability_mutation = probability_mutation
        self.probability_crossover = probability_crossover
        if selection_method[0] == "tournament":
            self.selection = selection.create_tournament(
                selection_method[1], self.minimize
            )
        else:
            self.selection = lambda r, ls, n: [x for x in ls[:n]]
        self.force_individual = force_individual

    def create_individual(self, depth: int):
        genotype = self.representation.create_individual(
            r=self.random, g=self.grammar, depth=depth
        )
        return Individual(
            genotype=genotype,
            fitness=None,
        )

    def evaluate(self, individual: Individual) -> float:
        if individual.fitness is None:
            phenotype = self.representation.genotype_to_phenotype(
                self.grammar, individual.genotype
            )
            individual.fitness = self.evaluation_function(phenotype)
        return individual.fitness

    def fitness_correction_for_depth(self, individual: Individual) -> float:
        if self.favor_less_deep_trees: # grammatical evolution does not have distance_to_term
            return individual.genotype.distance_to_term * 10 ** -25
        else:
            return 0

    def keyfitness(self):
        if self.minimize:
            return lambda x: self.evaluate(x) + self.fitness_correction_for_depth(x)
        else:
            return lambda x: -self.evaluate(x) - self.fitness_correction_for_depth(x)

    def evolve(self, verbose=0) -> Tuple[Individual, float, Any]:
        # TODO: This is not ramped half and half
        population = self.init_population()
        if self.force_individual is not None:
            population[0] = Individual(
                genotype=relabel_nodes_of_trees(
                    self.force_individual, self.grammar.non_terminals, self.max_depth
                ),
                fitness=None,
            )
        population = sorted(population, key=self.keyfitness())

        gen = 0
        start = time.time()

        while (not self.timer_stop_criteria and gen < self.number_of_generations) or (
            self.timer_stop_criteria and (time.time() - start) < 60
        ):
            npop = self.novelty(self.n_novelties)
            npop.extend(self.elitism(population, self.keyfitness()))
            spotsLeft = self.population_size - len(npop)
            for _ in range(spotsLeft // 2):
                candidates = self.selection(self.random, population, 2)
                (p1, p2) = candidates[0], candidates[1]
                if self.random.randint(0, 100) < self.probability_crossover * 100:
                    (p1, p2) = self.cross_over(p1, p2)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p1 = self.mutation(p1)
                if self.random.randint(0, 100) < self.probability_mutation * 100:
                    p2 = self.mutation(p2)
                npop.append(p1)
                npop.append(p2)

            population = npop
            population = sorted(population, key=self.keyfitness())

            time_gen = time.time() - start
            for cb in self.callbacks:
                cb.process_iteration(gen + 1, population, time_gen)

            if self.safe_gen_to_csv[0] != "":
                self.write_to_csv(
                    self.safe_gen_to_csv[0],
                    population,
                    (gen + 1),
                    (time.time() - start),
                    self.safe_gen_to_csv[1],
                )
            if verbose == 1:
                # self.printFitnesses(population, "G:" + str(gen))
                print("Best population:{}.".format(population[0]))
            if not self.timer_stop_criteria:
                print(
                    "BEST at",
                    gen + 1,
                    "/",
                    self.number_of_generations,
                    "is",
                    round(self.evaluate(population[0]), 4),
                )
            else:
                print(
                    "BEST at",
                    gen + 1,
                    "is",
                    round(self.evaluate(population[0]), 4),
                )
            if (
                self.target_fitness is not None
                and population[0].fitness == self.target_fitness
            ):
                break
            gen += 1
        self.final_population = population
        return (
            population[0],
            self.evaluate(population[0]),
            self.representation.genotype_to_phenotype(
                self.grammar, population[0].genotype
            ),
        )

    def init_population(self):
        return [
            self.create_individual(self.max_depth) for _ in range(self.population_size)
        ]

    def printFitnesses(self, pop, prefix):
        print(prefix)
        for x in pop:
            print(round(self.evaluate(x), 2), str(x))
        print("---")

        # "genotype_as_str",fitness_value,depth,number_of_the_generation,time_since_the_start_of_the_evolution

    def write_to_csv(
        self,
        file_name,
        population: List[Individual],
        number_of_the_generation,
        time_since_the_start_of_the_evolution,
        writing_method,
    ):
        if number_of_the_generation == 1:
            with open(f"{file_name}.csv", "w", newline="") as outfile:
                writer = csv.writer(outfile)
                writer.writerow(
                    [
                        "genotype_as_str",
                        "fitness",
                        "depth",
                        "number_of_the_generation",
                        "time_since_the_start_of_the_evolution",
                        "seed",
                    ]
                )

        csv_lines = list()
        if writing_method == "only_best_individual":
            population = [population[0]]
        for ind in population:
            genotype_as_str = str(ind.genotype)
            fitness = str(ind.fitness)
            if self.representation == treebased_representation:
                depth = ind.genotype.distance_to_term
            else:
                depth = -1
            csv_line = [
                genotype_as_str,
                fitness,
                depth,
                number_of_the_generation,
                time_since_the_start_of_the_evolution,
                self.seed,
            ]
            csv_lines.append(csv_line)

        with open(f"{file_name}.csv", "a", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerows(csv_lines)
