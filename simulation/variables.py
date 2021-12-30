from itertools import product

N_SIMULATIONS = 1000
NEEDLE = ("1c PieceRate", "4c PieceRate", "10c PieceRate")
N_TREATMENTS = (20, 50, 100)
STRATEGY = ("adaptive", "random")

variables = product(range(N_SIMULATIONS), NEEDLE, N_TREATMENTS, STRATEGY)
