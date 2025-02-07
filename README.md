# AI-Driven Derivation of Fundamental Physical Equations

This repository contains advanced Python implementations that use **Genetic Programming (GP)**—combined with reinforcement learning and rigorous dimensional analysis—to automatically rediscover fundamental physical laws. In particular, the scripts aim to derive:

- Einstein’s mass–energy equivalence formula: **E = m c²**
- The structure of the time-independent **Schrödinger equation**

Each script integrates a state-of-the-art symbolic regression framework with:
- A full candidate representation (as DAGs via DEAP)
- An advanced cost function that includes data fidelity, complexity regularization, and rigorous dimensional analysis
- A dimensional analysis module to recursively compute the physical dimensions of candidate expressions
- A reinforcement-learning (Q-learning) module to dynamically adapt the mutation rate during evolution

---

## Repository Contents

- **computational Formulation.py**  
  Contains the core, advanced implementation. This script demonstrates the full pipeline—from generating simulated data to evolving candidate expressions using GP with reinforcement learning and dimensional consistency checks. It serves as the foundation for the methods used in the other scripts.

- **Autonomous AI Derivation of E = mc2.py**  
  Implements a genetic programming approach to rediscover Einstein's mass–energy equivalence (E = m c²) using simulated data. It includes:
  - Random mass data generation and corresponding energy values (with Gaussian noise)
  - A primitive set of basic mathematical operations (including protected division and power)
  - A fitness function that combines data fidelity and a penalty for dimensional inconsistency
  - Evolutionary operators (selection, crossover, and mutation) to evolve candidate expressions

- **Derivation of Schrodinger’s Equation from First Principles.py**  
  Applies GP to autonomously derive the structure of the time-independent Schrödinger equation from simulated quantum data. This script:
  - Generates synthetic datasets for two scenarios: a free particle (V(x)=0) and a harmonic oscillator (V(x)=½ k x²)
  - Constructs a primitive set that includes the wavefunction (Ψ) and its derivatives (dΨ/dx, d²Ψ/dx²)
  - Evaluates candidate expressions by measuring their ability to satisfy the Schrödinger equation (minimizing the squared residual)
  - Incorporates a dimensional consistency check using Sympy to ensure physically meaningful expressions

---

## Dependencies

This project relies on the following Python libraries:
- [DEAP](https://github.com/DEAP/deap) – for genetic programming
- [NumPy](https://numpy.org/) – for numerical computations
- [Sympy](https://www.sympy.org/en/index.html) – for symbolic mathematics and dimensional analysis
- [Pandas](https://pandas.pydata.org/) – for data handling (used in the Schrödinger equation script)
- [Matplotlib](https://matplotlib.org/) – for plotting convergence graphs

Install the required packages using `pip`:

```bash
pip install deap numpy sympy pandas matplotlib
