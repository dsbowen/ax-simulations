from itertools import product

N_SIMULATIONS = 1000
NEEDLE = ("4c_PieceRate", "10c_PieceRate")
N_TREATMENTS = (40, 50, 60)
STRATEGY = ("adaptive", "random")

variables = product(range(N_SIMULATIONS), NEEDLE, N_TREATMENTS, STRATEGY)
