import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import PchipInterpolator
import seaborn as sns
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation
import tensorflow as tf
import time

def get_probs(state, modes):
    probs = {}
    for i in range(modes):
        basis = tuple(1 if j == i else 0 for j in range(modes))
        probs[basis] = state.ket()[*basis]
    return probs

def show_probs(state, modes):
    probs = get_probs(state, modes)
    for basis, state in probs.items():
        print(basis, state)

modes = 2
prog = sf.Program(modes)
cutoff = modes
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    
ket = np.zeros([cutoff]*modes, dtype=np.float32)
ket[(0,)*(modes-1) + (1,)] = 1.0 # 1 photon in the first mode so input state is (1,0,...0)

with prog.context as q:
    sf.ops.Ket(ket )| q
    BSgate(np.pi/4, np.pi) | (q[0], q[1])

results = eng.run(prog)
state = results.state
show_probs(state, modes)
print(state.ket())

# TODO:  measure the expected collision probability, which should be anticoncentrated for random circuits in log depth (https://arxiv.org/abs/2011.12277)