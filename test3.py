from functions import*
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from quimb.tensor.optimize import TNOptimizer
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
'''
Postselecting on random circuits
'''

avg_rnd = []
avg_rnd_pkd = []
delta = 0.003
peak = 0

n = 12 # system size
rqc_depth = 40

rqc = qmps_f(n, in_depth=rqc_depth, n_Qbit=n-1, seed_init=0, qmps_structure="brickwall",)

psi_rnd = abs((rqc^all).data.reshape(2**n))**2
psi_rnd = np.sort(psi_rnd)[::-1]

while peak < delta:
    pqc = qmps_f(n, in_depth=rqc_depth, n_Qbit=n-1, seed_init=0, qmps_structure="brickwall",)

    psi_pkd = abs((pqc^all).data.reshape(2**n))**2
    psi_pkd = np.sort(psi_pkd)[::-1]
    peak = psi_pkd[0]
    print(psi_pkd)

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ax = plt.figure(figsize=(5, 5)).gca()
ax.xaxis.set_major_locator(MultipleLocator(1000))
plt.plot(range(2**n), psi_rnd, label="Random output weight", color='blue', alpha=0.7)
plt.plot(range(2**n), psi_pkd, label="Peaked output weight", color='red', linestyle='dashed', alpha=0.7)
plt.xlabel("Bitstring Index")
plt.ylabel("Probability")
plt.title("output weight")
plt.legend()
plt.yscale("log")
plt.show() 
