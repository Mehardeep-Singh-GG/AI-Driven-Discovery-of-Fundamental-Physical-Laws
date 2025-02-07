import operator, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy
from deap import base, creator, gp, tools, algorithms
from multiprocessing import Pool


##     AI-Driven Derivation of Schrodinger’s Equation from First Principles    ##

# ----------------------------
# 1. Simulated Data Generation (Free Particle and Harmonic Oscillator)
# ----------------------------

def generate_schrodinger_data(num_samples=200, potential_type='free', noise_std=0.01, seed=42):
    """
    Generate synthetic data for the time-independent Schrodinger equation:
        - (hbar^2 / 2m) * d^2Psi/dx^2 + V(x) * Psi = E * Psi

    For the free particle, V(x) = 0. For harmonic oscillator, V(x) = 0.5 * k * x^2.
    """
    np.random.seed(seed)
    hbar = 1.0  # Planck's constant (reduced)
    m = 1.0  # Mass of the particle

    # Position values
    x_vals = np.linspace(-5, 5, num_samples)

    if potential_type == 'free':
        V = np.zeros_like(x_vals)
        Psi_true = np.exp(-x_vals ** 2)  # Gaussian wave packet as a simple solution
    elif potential_type == 'harmonic':
        k = 1.0  # Spring constant
        V = 0.5 * k * x_vals ** 2
        Psi_true = np.exp(-x_vals ** 2 / 2)  # Ground state wavefunction
    else:
        raise ValueError("Unknown potential type. Choose 'free' or 'harmonic'.")

    # Precompute first and second derivatives of Psi for GP terminals
    dPsi_dx = np.gradient(Psi_true, x_vals)
    d2Psi_dx2 = np.gradient(dPsi_dx, x_vals)

    # Add noise to the wavefunction
    noise = np.random.normal(0, noise_std * np.mean(np.abs(Psi_true)), size=num_samples)
    Psi_noisy = Psi_true + noise

    data = pd.DataFrame({'x': x_vals, 'V': V, 'Psi': Psi_noisy, 'dPsi_dx': dPsi_dx, 'd2Psi_dx2': d2Psi_dx2})
    return data


# Generate datasets
free_particle_data = generate_schrodinger_data(potential_type='free')
harmonic_data = generate_schrodinger_data(potential_type='harmonic')

# ----------------------------
# 2. Define the Primitive Set for GP
# ----------------------------

pset = gp.PrimitiveSet("MAIN", 4)  # Inputs: x, Psi, dPsi_dx, d2Psi_dx2
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='Psi')
pset.renameArguments(ARG2='dPsi_dx')
pset.renameArguments(ARG3='d2Psi_dx2')

# Basic operations
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)


# Protected division
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0


pset.addPrimitive(protectedDiv, 2)

# Ephemeral constant
pset.addEphemeralConstant("rand101", lambda: random.uniform(-2, 2))

# ----------------------------
# 3. Setup the Genetic Programming Framework
# ----------------------------

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Compile the individual into a callable function
toolbox.register("compile", gp.compile, pset=pset)

# ----------------------------
# 4. Define Dimensional Consistency Check
# ----------------------------

x_sym, Psi_sym, dPsi_dx_sym, d2Psi_dx2_sym = sympy.symbols('x Psi dPsi_dx d2Psi_dx2')


def check_dimensional_consistency(expr_str):
    """
    Check if the candidate expression is dimensionally consistent with the
    Schrodinger equation.
    """
    try:
        expr_sym = sympy.sympify(expr_str, locals={'x': x_sym, 'Psi': Psi_sym, 'dPsi_dx': dPsi_dx_sym,
                                                   'd2Psi_dx2': d2Psi_dx2_sym})
        # In Schrodinger equation, terms should have units of energy * Psi
        # Assume x has units of length (L), Psi is dimensionless, and d/dx adds L^-1
        # Return a penalty for dimensional mismatch
        terms = expr_sym.as_ordered_terms()
        for term in terms:
            if not term.has(d2Psi_dx2_sym) and not term.has(Psi_sym):
                return 1e5  # Penalize terms that do not match expected dimensions
        return 0
    except Exception:
        return 1e6  # Large penalty for non-parsable expressions


# ----------------------------
# 5. Define the Fitness Evaluation Function
# ----------------------------

def evaluate_schrodinger(individual, data):
    """
    Evaluate candidate expressions for the Schrodinger equation.
    The fitness measures how closely the expression satisfies the equation.
    """
    func = toolbox.compile(expr=individual)
    residuals = []

    for index, row in data.iterrows():
        x_val = row['x']
        Psi_val = row['Psi']
        dPsi_dx_val = row['dPsi_dx']
        d2Psi_dx2_val = row['d2Psi_dx2']
        V_val = row['V']

        # True Schrodinger equation residual: (-hbar^2/2m) * d2Psi_dx2 + V*Psi - E*Psi ~ 0
        try:
            residual = func(x_val, Psi_val, dPsi_dx_val, d2Psi_dx2_val)
            residuals.append(residual ** 2)
        except Exception:
            residuals.append(1e6)  # Penalize invalid expressions

    mse = np.mean(residuals)
    dim_penalty = check_dimensional_consistency(str(individual))
    return mse + dim_penalty,


# Parallel evaluation
toolbox.register("map", Pool().map)
toolbox.register("evaluate", evaluate_schrodinger, data=harmonic_data)

# ----------------------------
# 6. Genetic Operators and Evolutionary Algorithm Setup
# ----------------------------

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Control tree height
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


# ----------------------------
# 7. Run the Evolutionary Process with Cross-Validation
# ----------------------------

def run_evolution(data, generations=100, pop_size=300, cxpb=0.5, mutpb=0.2, seed=42):
    random.seed(seed)
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    # Split data into training and validation sets
    train_data = data.sample(frac=0.8, random_state=seed)
    val_data = data.drop(train_data.index)

    # Adjust fitness evaluation for training data
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluate_schrodinger, data=train_data)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=generations, stats=stats,
                                   halloffame=hof, verbose=True)

    # Validate the best individual
    best_ind = hof[0]
    val_fitness = evaluate_schrodinger(best_ind, val_data)

    print(f"Validation Fitness of Best Individual: {val_fitness[0]:.4f}")
    return pop, hof, log


# Run evolution
pop, hof, log = run_evolution(harmonic_data)

# ----------------------------
# 8. Results and Visualization
# ----------------------------

best_individual = hof[0]
print("\nBest individual found:")
print(best_individual)
print("Fitness:", best_individual.fitness.values[0])

# Visualization
gen = log.select("gen")
min_fitness = log.select("min")

plt.figure(figsize=(10, 5))
plt.plot(gen, min_fitness, label='Minimum Fitness')
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Convergence of the Evolutionary Process")
plt.legend()
plt.grid(True)
plt.show()
