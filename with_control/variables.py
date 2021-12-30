from itertools import product

N_SIMULATIONS = 1000
NEEDLE = ("4c_PieceRate", "10c_PieceRate")
N_TREATMENTS = (30, 40, 50)
STRATEGY = ("adaptive", "random")

variables = product(range(N_SIMULATIONS), NEEDLE, N_TREATMENTS, STRATEGY)
