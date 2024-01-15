import numpy
from agents import MACDAgent, RSIAgent, RSI_MACD_Agent



def pop_fitness(env, agents_pop):
    """
    calculate the population fitness which is the cumulative return
    :param env: the trading environment
    :param agents_pop: population of trading agent
    :return: ndarray of fitness value with the same length as agents_pop
    """
    all_fitness = []

    for agent in agents_pop:
        state = env.reset()
        done = 0
        while done == 0:
            action = agent.get_action(state)
            state, reward, done, info = env.step(action)
        cum_returns = env.get_cum_return()
        all_fitness.append(cum_returns)
    return numpy.array(all_fitness)


def select_parents(pop, fitness, num_parents):
    """
    Selecting the best individuals in the current generation as parents for producing the children of the next generation.
    :param pop: the current trading agent population
    :param fitness: ndarray of fitness value which correpond to the pop array
    :param num_parents: number of parents to select for producing the next generation
    :return: the selected parents (the best individual of the generation)
    """
    parents = []
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents.append(pop[max_fitness_idx])
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size, num_params):
    """
    produce children from parents
    :param parents: list of parents trading agent
    :param offspring_size: number of children to produce
    :param num_params: number of parameters of the agents
    :return: a ndarray of shape (offspring_size, num_params) which represents the children of the parents
    """
    offspring = numpy.empty((offspring_size, num_params), int)
    crossover_point = numpy.random.randint(0, num_params)
    for i in range(offspring_size):
        parent_indx1 = i % len(parents)
        parent_indx2 = (i+1) % len(parents)
        params = parents[parent_indx1].get_params()[:crossover_point] + parents[parent_indx2].get_params()[crossover_point:]
        offspring[i, :] = params
    return offspring

def mutation(offspring_crossover, num_weights, num_mutations=1, max_params_value=30):
    """
    apply random mutation on children
    :param offspring_crossover: array of shape (number_of_children, num_weights)
    :param num_weights: number of parameters of the given agent
    :param num_mutations: num of mutation to apply for each children
    :param max_params_value: max value for parameters
    :return: the new children with mutations
    """
    for idx in range(offspring_crossover.shape[0]):
        for mutation in range(num_mutations):
            param_idx = numpy.random.randint(0, num_weights)
            random_mut = numpy.random.randint(-3, 3)
            new_val = offspring_crossover[idx, param_idx] + random_mut
            if new_val <= 0:
                new_val = 1
            if new_val > max_params_value:
                new_val = max_params_value
            offspring_crossover[idx, param_idx] = new_val
    return offspring_crossover


def init_population_macd(pop_size, max_params_value):
    """
    init a random population for the macd agent
    :param pop_size: int population size
    :param max_params_value: int max parameters value
    :return: list of macd agent of size pop_size
    """
    parameters_windows = {"s_period": [5, max_params_value - 5], "l_period": [10, max_params_value], "signal_period":
        [5, max_params_value]}
    new_population = []
    for i in range(pop_size):
        s_period = numpy.random.randint(5, max_params_value - 10)
        l_period = numpy.random.randint(10, max_params_value)
        while s_period >= l_period:
            l_period = numpy.random.randint(10, max_params_value)
        signal_period = numpy.random.randint(5, max_params_value)
        new_population.append(MACDAgent(s_period, l_period, signal_period))
    return new_population

def init_population_rsi(pop_size, max_params_value):
    """
     init a random population for the rsi agent
    :param pop_size: int population size
    :param max_params_value: int max parameters value
    :return: list of rsi agent of size pop_size
    """
    parameters_windows = {"period": [5, max_params_value], "low_threshold": [5, 50],
                          "high_threshold": [51, 95]}
    new_population = []
    for i in range(pop_size):
        period = numpy.random.randint(5, max_params_value)
        low_threhold = numpy.random.randint(5, 50)
        high_threshold = numpy.random.randint(51, 95)
        new_population.append(RSIAgent(period, low_threhold, high_threshold))
    return new_population

def init_population_rsi_macd(pop_size, max_params_value):
    """
    init a random population for the rsi_macd agent
    :param pop_size: int population size
    :param max_params_value: int max parameters value
    :return: list of rsi_macd agent of size pop_size
    """
    parameters_windows = {"period": [5, max_params_value], "low_threshold": [5, 50],
                          "high_threshold": [51, 95],"s_period": [5, max_params_value - 5],
                          "l_period": [10, max_params_value], "signal_period": [5, max_params_value]}
    new_population = []
    for i in range(pop_size):
        rsi_period = numpy.random.randint(5, max_params_value)
        rsi_low_threhold = numpy.random.randint(5, 50)
        rsi_high_threshold = numpy.random.randint(51, 95)
        s_period = numpy.random.randint(5, max_params_value - 10)
        l_period = numpy.random.randint(10, max_params_value)
        while s_period >= l_period:
            l_period = numpy.random.randint(10, max_params_value)
        signal_period = numpy.random.randint(5, max_params_value)
        new_population.append(RSI_MACD_Agent(rsi_period, rsi_low_threhold, rsi_high_threshold,
                                             s_period, l_period, signal_period))
    return new_population

def gen_algo(env, num_params, init_pop, agent_constructor, num_generations=50, pop_size=50, num_parents=25,
             num_mutations=0, max_params_value=50):
    """
    Apply genetic algorithm to find best trading agent.
    :param env: a trading environment
    :param num_generations: number of generation
    :param pop_size: population size at each generation
    :param num_parents: number of parents to keep at each generation
    :param num_params: number of parameters of the given agent
    :param init_pop: function to initialize a random agent population
    :param agent_constructor: constuctor function for creating the agent
    :param num_mutations: number of mutation at each generation for each children
    :param max_params_value: int max parameters value
    :return: the best agent (with the best parameters) on the given environment
    """

    # Creating the initial population
    new_population = init_pop(pop_size, max_params_value)
    for generation in range(num_generations):
        print("Generation : ", generation)

        # Measuring the fitness
        fitness = pop_fitness(env, new_population)
        print("Fitness")
        print(fitness)

        # Selecting the best parents to produce children
        parents = select_parents(new_population, fitness,
                                 num_parents)

        # Generating children
        children_crossover = crossover(parents, offspring_size=pop_size - len(parents),num_params=num_params)

        # Adding some mutation to the
        if num_mutations:
            children_crossover = mutation(children_crossover, num_weights=num_params, num_mutations=num_mutations, max_params_value=max_params_value)

        # creating agent from the parameters
        children_crossover = [agent_constructor(*children_crossover[i, :]) for i in range(children_crossover.shape[0])]

        # reset parents agent
        for parent in parents:
            parent.reset()

        # Creating the new population based on the parents and children
        new_population[0:len(parents)] = parents
        new_population[len(parents):] = children_crossover

    # Selecting the best agent with the best fitness
    fitness = pop_fitness(env, new_population)
    best_match_idx = numpy.where(fitness == numpy.max(fitness))[0][0]

    print("Best solution : ", new_population[best_match_idx])
    print("Best solution fitness : ", fitness[best_match_idx])

    return new_population[best_match_idx].reset()
