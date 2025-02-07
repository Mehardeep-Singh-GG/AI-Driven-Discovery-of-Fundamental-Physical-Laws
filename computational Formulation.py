import random
import operator
import math
import numpy as np
import sympy as sp

from deap import base, creator, gp, tools

# =============================================================================
# 1. SET UP THE SYMBOLIC REGRESSION FRAMEWORK
# =============================================================================

# Create a primitive set for one dummy input (x). (x is not used but required.)
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')

# Basic arithmetic primitives
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)


# Protected division to avoid division by zero.
def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1


pset.addPrimitive(protectedDiv, 2)

# Add some transcendentals (these require dimensionless arguments)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# Ephemeral constants (random constants, assumed dimensionless)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))

# Terminals representing fundamental physical quantities.
# We wish the system to “discover” the relation E = m * c**2.
pset.addTerminal("m")
pset.addTerminal("c")

# =============================================================================
# 2. DIMENSIONAL ANALYSIS MODULE
# =============================================================================
# Define the mapping for our base terminals:
# m has dimension [M] = M^1
# c has dimension [L T^-1] i.e. {L: 1, T: -1}
DIM_MAP = {
    'm': {'M': 1, 'L': 0, 'T': 0},
    'c': {'M': 0, 'L': 1, 'T': -1},
}


# Numbers and ephemeral constants are assumed dimensionless.

def add_dims(dim1, dim2):
    """Add dimensions when multiplying: exponents add."""
    result = {}
    for key in set(dim1.keys()).union(dim2.keys()):
        result[key] = dim1.get(key, 0) + dim2.get(key, 0)
    return result


def sub_dims(dim1, dim2):
    """Subtract dimensions when dividing: exponents subtract."""
    result = {}
    for key in set(dim1.keys()).union(dim2.keys()):
        result[key] = dim1.get(key, 0) - dim2.get(key, 0)
    return result


def mul_dims(dim, exponent):
    """Multiply dimension exponents by a constant exponent."""
    return {key: exponent * val for key, val in dim.items()}


def compute_dimension(expr):
    """
    Recursively compute the dimension of a sympy expression.
    Returns a dictionary of exponents for base units {M, L, T} or None if inconsistent.
    """
    # If expr is a number, assume dimensionless.
    if expr.is_Number:
        return {}
    # If expr is a symbol, look up its dimension.
    elif expr.is_Symbol:
        return DIM_MAP.get(expr.name, {})  # Default to dimensionless if unknown.
    # Addition/Subtraction: all terms must have the same dimension.
    elif expr.is_Add:
        dims = [compute_dimension(arg) for arg in expr.args]
        if any(d is None for d in dims):
            return None
        first = dims[0]
        for d in dims[1:]:
            if d != first:
                return None  # Inconsistent dimensions
        return first
    # Multiplication: dimensions add.
    elif expr.is_Mul:
        dims = {}
        for arg in expr.args:
            d = compute_dimension(arg)
            if d is None:
                return None
            dims = add_dims(dims, d)
        return dims
    # Power: if the exponent is a number, multiply base dimension.
    elif expr.is_Pow:
        base, exponent = expr.args
        base_dim = compute_dimension(base)
        if base_dim is None:
            return None
        if exponent.is_Number:
            return mul_dims(base_dim, float(exponent))
        else:
            # Exponent not a constant → cannot reliably compute dimension.
            return None
    # Functions: sin, cos, tan, exp, log require a dimensionless argument and return dimensionless.
    elif expr.func in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]:
        arg_dim = compute_dimension(expr.args[0])
        if arg_dim is None or any(val != 0 for val in arg_dim.values()):
            return None
        return {}
    else:
        # For any other function, assume dimensionless.
        return {}


def dimensional_penalty(sym_expr, target_dim):
    """
    Compute a penalty based on the difference between the candidate's computed dimension
    and the target dimension (given as a dictionary of exponents).
    If dimensions cannot be computed (i.e. inconsistent), a high penalty is returned.
    """
    expr_dim = compute_dimension(sym_expr)
    if expr_dim is None:
        return 100.0  # Heavy penalty for inconsistency.
    penalty = 0.0
    # Consider all base units present in either.
    for key in set(target_dim.keys()).union(expr_dim.keys()):
        diff = abs(expr_dim.get(key, 0) - target_dim.get(key, 0))
        penalty += diff
    return penalty


# Target dimension for energy: [E] = M * L^2 * T^-2
TARGET_DIM = {'M': 1, 'L': 2, 'T': -2}

# =============================================================================
# 3. GENETIC PROGRAMMING SETUP
# =============================================================================

# Define the fitness (minimization) and individual (a GP tree)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Generate expression trees (using half-and-half initialization)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# =============================================================================
# 4. UTILITY FUNCTIONS: COMPILATION, EVALUATION, AND COST FUNCTION
# =============================================================================

def safe_compile(individual):
    """
    Converts a DEAP GP individual into a Sympy expression.
    """
    expr_str = str(individual)
    # Create a local dictionary mapping the terminals.
    local_dict = {'m': sp.symbols('m'), 'c': sp.symbols('c'), 'x': sp.symbols('x')}
    try:
        sym_expr = sp.sympify(expr_str, locals=local_dict)
    except Exception as e:
        # If conversion fails, return a dummy value.
        sym_expr = sp.sympify("0")
    return sym_expr


def evaluate_individual(individual, data, lambda_reg=0.01, nu_dim=1.0):
    """
    Evaluate a candidate expression using:
      1. Mean Squared Error (MSE) on the provided data.
      2. A complexity penalty (proportional to the number of nodes in the GP tree).
      3. A dimensional consistency penalty (using our advanced dimensional analysis).

    The dataset 'data' is a list of tuples (m_value, c_value, target)
    where target is defined as m * c**2 (the energy).
    """
    sym_expr = safe_compile(individual)
    m_sym, c_sym, x_sym = sp.symbols('m c x')
    try:
        func = sp.lambdify((m_sym, c_sym, x_sym), sym_expr, modules=['math'])
    except Exception:
        return 1e6,

    errors = []
    for (m_val, c_val, target) in data:
        try:
            pred = func(m_val, c_val, 0)  # x is a dummy argument.
            errors.append((pred - target) ** 2)
        except Exception:
            errors.append(1e6)
    mse = np.mean(errors)
    complexity = len(individual)
    dim_pen = dimensional_penalty(sym_expr, TARGET_DIM)

    total_cost = mse + lambda_reg * complexity + nu_dim * dim_pen
    return total_cost,


# Prepare a dummy dataset for the target relation E = m * c**2.
dummy_data = [(1, 3, 9), (2, 3, 18), (3, 3, 27)]
toolbox.register("evaluate", evaluate_individual, data=dummy_data)

# Genetic operators: selection, crossover, and mutation.
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# =============================================================================
# 5. REINFORCEMENT LEARNING MODULE FOR HYPERPARAMETER TUNING
# =============================================================================
# We define a small set of mutation rates and update Q-values (action-value estimates)
# via Q-learning to dynamically choose the mutation rate.

mutation_rates = [0.05, 0.1, 0.2]
q_values = {mr: 0.0 for mr in mutation_rates}
current_mutation_rate = 0.1  # initial mutation rate

# RL hyperparameters
alpha_rl = 0.5  # learning rate
gamma_rl = 0.9  # discount factor


def select_mutation_rate(reward):
    """
    Select a mutation rate using a simple epsilon-greedy Q-learning update.
    The Q-value for the previously chosen mutation rate is updated based on the received reward,
    then the best (or a random exploratory) mutation rate is selected.
    """
    global current_mutation_rate
    epsilon = 0.1  # exploration probability

    # Epsilon-greedy: choose a random rate with probability epsilon.
    if random.random() < epsilon:
        action = random.choice(mutation_rates)
    else:
        action = max(q_values, key=q_values.get)

    # Q-learning update for the previous action.
    q_values[current_mutation_rate] = q_values[current_mutation_rate] + \
                                      alpha_rl * (reward + gamma_rl * q_values[action] - q_values[
        current_mutation_rate])

    current_mutation_rate = action
    return action


def rl_mutate(individual):
    """
    Mutation operator that uses the mutation rate selected by the RL agent.
    """
    # Get an updated mutation rate (the reward is provided externally in the evolutionary loop)
    mr = select_mutation_rate(reward=0)
    mutant, = toolbox.mutate(individual, prob=mr)
    return mutant


toolbox.register("mutate_rl", rl_mutate)


# =============================================================================
# 6. MAIN EVOLUTIONARY LOOP
# =============================================================================

def main():
    random.seed(42)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)  # Hall-of-Fame to track the best candidate

    # Statistics for reporting.
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    ngen = 20  # number of generations
    cxpb = 0.5  # crossover probability

    for gen in range(ngen):
        # Selection (tournament selection) and cloning.
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover.
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation using our RL-based mutation operator.
        for i in range(len(offspring)):
            offspring[i] = toolbox.mutate_rl(offspring[i])
            del offspring[i].fitness.values

        # Evaluate individuals with invalid fitness.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Compute the average fitness (lower is better).
        avg_fitness = np.mean([ind.fitness.values[0] for ind in pop])
        # Define a reward signal (here: negative average cost gives a positive reward).
        reward = -avg_fitness

        # Update the RL agent's mutation rate selection.
        selected_mut_rate = select_mutation_rate(reward)

        # Report generation statistics.
        best_fit = min(ind.fitness.values[0] for ind in pop)
        print("Generation {}: Best fitness = {:.2f}, Mutation rate = {:.3f}".format(
            gen, best_fit, selected_mut_rate))

        hof.update(pop)

    print("\n=== Final Result ===")
    best_ind = hof[0]
    print("Best individual (GP tree):", best_ind)
    print("Fitness:", best_ind.fitness.values)
    best_expr = safe_compile(best_ind)
    print("Derived expression (sympy format):", best_expr)
    # Also show the computed dimension and its distance from target:
    computed_dim = compute_dimension(best_expr)
    print("Computed dimension:", computed_dim)
    print("Dimensional penalty:", dimensional_penalty(best_expr, TARGET_DIM))



