import numpy as np 
import quimb as qu
from quimb.tensor.optimize import TNOptimizer
from quimb.tensor import MatrixProductState
from functions import*
'''
Issue with norm_f while doing test.py
'''
def debug(name,x):
    print(name,type(x), x.shape)

def apply_gate(net, idx, q0,q1):
    U = net.tensors[idx].data.reshape(4,4)
    # U = qu.qarray(np.conj(U.T))
    net.gate_(U, (q0, q1), tags={'U'})

def rearrange_gates(n, depth, gates):
    shift= 1-depth%2
    res = []
    c = 0
    gates.reverse()
    for d in range(depth):
        slce = gates[c:c+(n-shift)//2]
        slce.reverse()
        res.extend(slce)
        c += (n-shift)//2
        shift = 1-shift
    return res

n=12
depth = n//2
rqc = qmps_f(n, in_depth=40, n_Qbit=n-1, seed_init=0, qmps_structure="brickwall",)
pqc = qmps_f(n, in_depth=40, n_Qbit=n-1, seed_init=0, qmps_structure="brickwall",)

optmzr = TNOptimizer(
    pqc,                                # our initial input, the tensors of which to optimize
    loss_fn=negative_overlap,
    norm_fn=norm_f,
    loss_target=-1+1e-2,
    constant_tags=['MPS'],
    loss_constants={'target': rqc},  # this is a constant TN to supply to loss_fn
    autodiff_backend='tensorflow',      # {'jax', 'tensorflow', 'autograd'}
    optimizer='L-BFGS-B',               # supplied to scipy.minimize
)
mps_opt = optmzr.optimize(500,tol = 1e-10,) # perform ~100 gradient descent steps'''

print("rqc",rqc)
print("mps_opt",mps_opt)
gates = mps_opt.tensors[n:]
rqc_combined = rqc
for gate in gates:
    rqc_combined = rqc_combined | gate.H
rqc_combined = MatrixProductState.from_TN(rqc_combined, site_ind_id="k{}", site_tag_id="I{}", L=rqc.L, cyclic=False)


psi_rnd = abs((rqc^all).data.reshape(2**n))**2
psi_pkd = abs((rqc_combined^all).data.reshape(2**n))**2
psi_rnd = np.sort(psi_rnd)[::-1]
psi_pkd = np.sort(psi_pkd)[::-1]


import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
ax = plt.figure(figsize=(5, 5)).gca()
ax.xaxis.set_major_locator(MultipleLocator(1000))
plt.plot(range(2**n), psi_rnd, label="Random output weight", color='teal', linestyle='-') #  marker='o', 
plt.plot(range(2**n), psi_pkd, label="Peaked output weight", color='olive', linestyle='-') #  marker='^',
plt.xlabel("Bitstring Index")
plt.ylabel("Probability")
plt.title("output weight")
plt.legend()
plt.yscale("log")
plt.show() 

# for i, tensor in enumerate(mps_opt.tensors):
#     print(f"Tensor {i}: shape {tensor.data.shape}, tags {tensor.tags}")
#     if tensor.data.shape != (2,2,2,2): 
#         print(tensor.data)
#     if tensor.data.shape == (2,2,2,2): 
#         U = tensor.data.reshape(4,4)
#         print((U@np.conj(U.T)).round(10))

# indices = rqc.tensors[-1].inds  # Extract the indices
# new_indices = {index: f"new_{index}" for index in indices if index.startswith("_")}  # Change only unnamed ones
# a = rqc.tensors[-1].reindex(new_indices)

# ctr = 0
# for r in range(depth): # pqc depth
#     if (depth-r)%2==1:
#         for i in range(0, n-1, 2):
#             U = gates[ctr].data.reshape(4,4)
#             U = qu.qarray(np.conj(U.T))
#             mps_opt.gate_(U, (i, i + 1), tags={'U'})
#             ctr += 1
#     else:
#         for i in range(1, n-1, 2):
#             U = gates[ctr].data.reshape(4,4)
#             U = qu.qarray(np.conj(U.T))
#             mps_opt.gate_(U, (i, i + 1), tags={'U'})
#             ctr += 1
