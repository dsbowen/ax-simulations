from itertools import product

N_SIMULATIONS = 1000
NEEDLE = ("1c_PieceRate",)
HAYSTACK = ("No_Payment",)
N_TREATMENTS = (20, 30, 40, 50, 60, 70)

variables = product(range(N_SIMULATIONS), NEEDLE, HAYSTACK, N_TREATMENTS)
