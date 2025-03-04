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

modes = 5
prog = sf.Program(modes)
cutoff = modes + 1
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

ket = np.zeros([cutoff]*modes, dtype=np.float32)
ket[(1,) + (0,)*(modes-1)] = 1.0 # 1 photon in the first mode so input state is (1,0,...0)

with prog.context as q:
    start = time.time()
    sf.ops.Fock(1) | q[0]
    BSgate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)) | (q[1], q[1 + 1])

results = eng.run(prog)
print("Fock(1) time:", time.time() - start)


if eng.run_progs:
    eng.reset()

prog = sf.Program(modes)
cutoff = modes + 1
eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

with prog.context as q:
    start = time.time()
    sf.ops.Ket(ket) | q
    BSgate(np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)) | (q[1], q[1 + 1])

results = eng.run(prog)
print("Ket(ket) time:", time.time() - start)