import numpy as np
import matplotlib.pyplot as plt

'''
Why is single-photon setup at small mode values throwing an error/slow compared to RCS?
A: Fock backend.
'''

def order(probs):
    state_probs = []
    for idx in np.ndindex(probs.shape):
        prob = probs[idx].item()
        if prob > 0:
            state = f"|{','.join(map(str, idx))}⟩"
            state_probs.append((state, prob))

    for state, prob in state_probs: 
        print(f"State {state}: {prob:.6f}")
    state_probs.sort(key=lambda x: x[1], reverse=True) # sort probabilities in descending order
    if len(state_probs) == 0:
        print("oops, no state probs!")
        exit()
    return range(1, len(state_probs) + 1), tuple(zip(*state_probs))[1]

def get_probs(modes, state):
    probs = []
    for i in range(modes):
        basis = tuple(1 if j == i else 0 for j in range(modes))
        probs.append(state.fock_prob(basis))
    return np.array(probs)

import strawberryfields as sf
from strawberryfields.ops import *

# Number of modes
m = 5
boson_sampling = sf.Program(m)
eng = sf.Engine("fock", backend_options={"cutoff_dim": 2})

with boson_sampling.context as q:
    
    # sf.ops.Ket(self.ket) | q
    Fock(1) | q[0]
    # self.apply_layer(q, record_gate_seq)
    BSgate(np.pi/4,np.pi/2) | (q[0],q[1])

results = eng.run(boson_sampling)
rqc = results.state
rqc.all_fock_probs()
data = (range(1, m + 1), get_probs(m, rqc)[::-1])
print(data[1],sum(data[1]))
ax = plt.figure(figsize=(5, 5)).gca()
ax.plot(*data, marker="^")
plt.show()