import operator, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy

# DEAP imports for genetic programming
from deap import base, creator, gp, tools

# ----------------------------
# 1. Simulated Data Generation
# ----------------------------

# Global constant: speed of light
c_val = 3e8


def generate_simulated_data(num_samples=200, noise_std=0.05, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # Generate random masses between 0.1 and 10 kg
    m = np.random.uniform(0.1, 10, size=num_samples)
    E_true = m * (c_val ** 2)  # True relation: E = m * c^2
    noise = np.random.normal(0, noise_std * np.mean(E_true), size=num_samples)
    E_noisy = E_true + noise
    data = pd.DataFrame({'m': m, 'E': E_noisy})
    return data


data = generate_simulated_data()
print("Sample of training data:")
print(data.head())

# ----------------------------
# 2. Define the Primitive Set for GP
# ----------------------------

# Create a primitive set for expressions with two inputs: m and c.
pset = gp.PrimitiveSet("MAIN", 2)  # 2 input variables: m, c
pset.renameArguments(ARG0='m')
pset.renameArguments(ARG1='c')

# Add basic arithmetic operators
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)


# Protected division to avoid division-by-zero errors
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


pset.addPrimitive(protectedDiv, 2)


# A protected power function: We restrict the power to avoid huge numbers.
def protectedPow(base, exponent):
    # Limit the exponent to a reasonable range.
    try:
        # If exponent is near an integer, round it.
        exponent = round(exponent, 2)
        # Avoid complex numbers or huge values
        result = math.pow(abs(base), exponent)
        # Restore the sign if base is negative and exponent is an integer
        if base < 0 and exponent % 1 == 0:
            result = -result
        return result
    except Exception:
        return 1.0


pset.addPrimitive(protectedPow, 2)

# Ephemeral constant: random float in [-2, 2]
pset.addEphemeralConstant("rand101", lambda: random.uniform(-2, 2))

# ----------------------------
# 3. Setup the Genetic Programming Framework
# ----------------------------

# We aim to minimize the fitness (error + penalty)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Register the expression generator (full method)
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)

# Structure initializers: individuals and population
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ----------------------------
# 4. Define the Dimensional Consistency Check
# ----------------------------

# We use sympy to attempt to extract the net exponents on m and c.
# For a candidate expression to be dimensionally consistent with energy,
# we expect an expression of the form: constant * m^1 * c^2.
m_sym, c_sym = sympy.symbols('m c')


def get_dimension_exponents(expr_str):
    """
    Convert the candidate expression (string) into a sympy expression
    and try to extract the net exponents on m and c.

    Returns (exp_m, exp_c) if successful. Otherwise, returns (None, None)
    and will trigger a penalty.
    """
    try:
        expr_sym = sympy.sympify(expr_str, locals={'m': m_sym, 'c': c_sym, 'protectedDiv': sympy.Div})
        # Expand the expression and collect powers of m and c.
        expr_sym = sympy.expand(expr_sym)
        # Assume expression is a product of a constant and powers of m and c.
        # For simplicity, we try to factor the expression.
        factors = sympy.factor(expr_sym)
        # Use sympy's as_coeff_mul to get the constant and the multiplicative factors.
        coeff, factors_tuple = factors.as_coeff_mul()
        exp_m = 0
        exp_c = 0
        # Loop over factors and sum up exponents on m and c.
        for factor in factors_tuple:
            # For example, factor might be m**alpha or c**beta.
            if factor.has(m_sym):
                # Get the exponent of m in this factor.
                exp = sympy.degree(factor, m_sym)
                exp_m += exp
            if factor.has(c_sym):
                exp = sympy.degree(factor, c_sym)
                exp_c += exp
        return exp_m, exp_c
    except Exception as e:
        # In case of any error, return None which triggers a heavy penalty.
        return None, None


def dimensional_penalty(individual):
    """
    Compute a penalty based on the deviation of the candidate's effective exponents
    from the expected ones: m exponent should be 1 and c exponent should be 2.
    If we cannot extract exponents, return a large penalty.
    """
    expr_str = str(individual)
    exp_m, exp_c = get_dimension_exponents(expr_str)
    if exp_m is None or exp_c is None:
        return 1e6  # large penalty for expressions that do not simplify
    # Compute squared deviation
    penalty = (exp_m - 1) ** 2 + (exp_c - 2) ** 2
    # Scale the penalty factor (tune as needed)
    return penalty * 1e5


# ----------------------------
# 5. Define the Fitness Evaluation Function
# ----------------------------

def evalSymbolicRegression(individual, data):
    """
    Evaluate the candidate expression on the training data.

    The total fitness is the mean squared error (MSE) on the data plus
    a penalty for dimensional inconsistency.
    """
    # Transform the tree expression into a callable function
    func = toolbox.compile(expr=individual)

    # Compute prediction error over the training data
    errors = []
    for index, row in data.iterrows():
        m_val = row['m']
        # Use the global c constant (could also use a variable, but here c is fixed)
        try:
            pred = func(m_val, c_val)
            errors.append((pred - row['E']) ** 2)
        except Exception:
            errors.append(1e6)
    mse = np.mean(errors)

    # Compute dimensional penalty
    dim_pen = dimensional_penalty(individual)

    return mse + dim_pen,


toolbox.register("evaluate", evalSymbolicRegression, data=data)
toolbox.register("compile", gp.compile, pset=pset)

# ----------------------------
# 6. Genetic Operators and Evolutionary Algorithm Setup
# ----------------------------

# Crossover and mutation operators
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Limit the height of the individuals for manageability
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# ----------------------------
# 7. Run the Evolutionary Process
# ----------------------------

def run_evolution(generations=50, pop_size=300, cxpb=0.5, mutpb=0.2, seed=42):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=generations, stats=mstats,
                                   halloffame=hof, verbose=True)
    return pop, hof, log


# Import the evolution algorithm function from DEAP
from deap import algorithms

pop, hof, log = run_evolution(generations=100, pop_size=300)

# ----------------------------
# 8. Results and Interpretation
# ----------------------------

best_individual = hof[0]
print("\nBest individual found:")
print(best_individual)
print("Fitness:", best_individual.fitness.values[0])

# Attempt to extract and display the candidate expression in human-readable form
expr_str = str(best_individual)
print("\nDerived Expression:")
print(expr_str)

# Use sympy to simplify and display the effective exponents (if possible)
exp_m, exp_c = get_dimension_exponents(expr_str)
if exp_m is not None and exp_c is not None:
    print(f"Extracted exponents: m^{exp_m}, c^{exp_c}")
else:
    print("Could not extract dimensional exponents from the expression.")

# ----------------------------
# 9. Visualize Convergence (Optional)
# ----------------------------

gen = log.select("gen")
min_fitness = log.select("min")
plt.plot(gen, min_fitness, marker='o')
plt.xlabel("Generation")
plt.ylabel("Minimum Fitness (MSE + Dimensional Penalty)")
plt.title("Convergence over Generations")
plt.show()
