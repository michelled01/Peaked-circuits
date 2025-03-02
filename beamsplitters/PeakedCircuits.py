#!/usr/bin/env python
# coding: utf-

import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns
import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
colors = sns.color_palette("rocket",6)
cc = itertools.cycle(colors)
plt.rcParams["font.family"] = "Cambria"

# From https://pennylane.ai/qml/demos/tutorial_haar_measure
def qr_haar(N):
    """Generate a Haar-random matrix with N modes using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B

    # Step 2
    Q, R = np.linalg.qr(Z)

    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])

    # Step 4
    return np.dot(Q, Lambda)


def pollman_like(q, modes, depth, optimize=False, passive_sd=0.1):
    if modes==0: return
    params = []
    for r in range(depth):
        print(r)
        if r % 2 == 0:
            for i in range(modes-1):
                if optimize:
                    theta = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    phi = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    params.append(theta)
                    params.append(phi)
                else:
                    theta = np.random.uniform(0, np.pi / 2)
                    phi = np.random.uniform(0, 2 * np.pi)
                BSgate(theta, phi) | (q[i], q[i + 1])
        else:
            for i in range(modes):
                if optimize:
                    theta = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    params.append(theta)
                else:
                    theta = np.random.uniform(0, 2 * np.pi)
                Rgate(theta) | q[i]
    return params


def brickwall_like(q, modes, depth, optimize=False, passive_sd=0.1):
    if modes==0: return
    params = []
    for r in range(depth):
        i = modes
        while i > 1:
            if (np.random.rand() < 0.5):
                if optimize:
                    theta = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    phi = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    params.append(theta)
                    params.append(phi)
                else:
                    theta = np.random.uniform(0, np.pi / 2)
                    phi = np.random.uniform(0, 2 * np.pi)
                BSgate(theta, phi) | (q[modes-i], q[modes-i + 1])
                i -= 2
            else:
                if optimize:
                    theta = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                    params.append(theta)
                else:
                    theta = np.random.uniform(0, 2 * np.pi)
                Rgate(theta) | q[modes-i] 
                i -= 1
        if i == 1:
            if optimize:
                theta = tf.Variable(tf.random.normal(shape=[], stddev=passive_sd))
                params.append(theta)
            else:
                theta = np.random.uniform(0, 2 * np.pi)
            Rgate(theta) | q[modes-i]
    return params


def boson_sampling(modes=5, depth=40, network="brickwall", sample=False):
    prog = sf.Program(modes)
    cutoff = modes + 1
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})

    ket = np.zeros([cutoff]*modes, dtype=np.float32)
    ket[(1,) + (0,)*(modes-1)] = 1.0 # 1 photon in the first mode so input state is (1,0,...0)

    U = qr_haar(modes)
    
    with prog.context as q:
        sf.ops.Ket(ket) | q
        # for i in range(1):
        #     Fock(1) | q[i]

        Interferometer(U) | q

        # if network == "brickwall":
        #     brickwall_like(q, modes, depth)
        # else:
        #     pollman_like(q, modes, depth)

        if (sample == True):
            MeasureFock() | q
            eng = sf.Engine("fock", backend_options={"cutoff_dim": + 1})
            results = eng.run(prog, run_options={"shots": 5000})
            samples = results.samples
            return samples

    results = eng.run(prog)
    assert results.state.is_pure==True # local preparation will produce a non-pure state
    return results.state, U

'''
another idea for sgd is to directly target the amplitude of the first mode (wlog bc U is haar-random).
'''
def boson_sampling_with_SGD(modes=5, depth=10, network='brickwall',
                            target=None, U=None, steps=50, learning_rate=0.1):
    '''
    sgd that learns the output distribution. inversions kick in around >= 4/5 * m peaking gates.
    any less => tries to learn the entire distribution rather than peak the first weight.
    '''
    fid_progress = []
    passive_sd = 0.1
    cutoff = modes + 1
    depth = 1 #modes//2
    t = 4*(modes-1)//5
     
    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(modes)

    bs_theta = tf.random.normal(shape=[depth], stddev=passive_sd) 
    bs_phi = tf.random.normal(shape=[depth], stddev=passive_sd)
    weights = tf.convert_to_tensor([bs_theta, bs_phi]*t)
    weights = tf.Variable(tf.transpose(weights))
    
    tf_params = []
    names = ["bs{}".format(i) for i in range(2*t)]
    for i in range(depth):
        tf_params_names = ["{}_{}".format(n, i) for n in names]
        tf_params.append(prog.params(*tf_params_names))
    tf_params = np.array(tf_params)

    assert tf_params.shape == weights.shape # (depth, modes) <-- modes//2 gates so modes paramters

    ket = np.zeros([cutoff] * modes, dtype=np.float32)
    ket[(1,) + (0,)*(modes-1)] = 1.0

    with prog.context as q:
        sf.ops.Ket(ket) | q

        # trainable peaking layer
        for l in range(depth):
            for i in range(t):
                BSgate(tf_params[l][2*i], tf_params[l][2*i+1]) | (q[0], q[i+1])

    # from tensorflow import keras
    # kl = keras.losses.KLDivergence()
    
    def cost(weights):
        mapping = {p.name: w for p, w in zip(tf_params.flatten(), tf.reshape(weights, [-1]))}
        state = eng.run(prog, args=mapping).state
        ket = state.ket()
        fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target)) ** 2

        cost = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target) - 1)
        return cost, fidelity, ket

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(50):
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss, fid, ket = cost(weights)
        
        fid_progress.append(fid.numpy())

        gradients = tape.gradient(loss, weights)
        opt.apply_gradients(zip([gradients], [weights]))

        print("Rep: {} Cost: {:.4f} Fidelity: {:.4f}".format(step, loss, fid))

    # plt.figure()
    # plt.plot(fid_progress)
    # plt.ylabel("Fidelity")
    # plt.xlabel("Step")
    # plt.show()

    optimized_parameters = np.array([p.numpy() for p in weights]).tolist()
    return run_inverse(modes, U, t, optimized_parameters)
     

def run_inverse(modes, U, t, opt_params):
    depth, _ = len(opt_params), len(opt_params[0])
    cutoff = modes + 1

    final_prog = sf.Program(modes)
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

    ket = np.zeros([cutoff] * modes, dtype=np.float32)
    ket[(1,) + (0,)*(modes-1)] = 1.0
    rev_params = []
    for p in reversed(opt_params):
        rev_params.append(p[::-1])

    with final_prog.context as q:
        sf.ops.Ket(ket) | q
        Interferometer(U) | q

        for l in range(depth):
            for i in range(t):
                BSgate(rev_params[l][2*i+1], rev_params[l][2*i]).H | (q[0], q[t-i])
    # final_prog.print()
    state = eng.run(final_prog).state
    return state


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
    return range(1, len(state_probs) + 1), tuple(zip(*state_probs))[1]

def print_distributions(parameters):
    ax = plt.figure(figsize=(5, 5)).gca()
    l = []
    rqc = boson_sampling(*parameters)
    pqc = boson_sampling_with_SGD(*parameters, target=rqc[0].ket(), U=rqc[1])
    r_axes = order(rqc[0].all_fock_probs())
    p_axes = order(pqc.all_fock_probs())
    idn = {'random layers': [r_axes, "o"], 'random + peaking layers': [p_axes, "^"]}
    for label, (data, marker) in idn.items():
        l.append(ax.plot(*data, color=next(cc), marker=marker, label=label)[0])

    ax.set_xlabel("single-photon modes", fontsize=10)
    ax.set_title("Output weight", fontsize=11)
    legend = ax.legend(l, idn.keys(), loc="upper right", frameon=True)
    
    ax.grid(visible=False)
    plt.tight_layout(pad=1)
    plt.show()
    ax.add_artist(legend)


def avg_peak_weight(modes=5, depth=40, shots=100, network="brickwall"):
    peak = []
    for i in range(shots):
        probs = boson_sampling(modes, depth, network, sample=False)
        peak.append(np.max(probs))

    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak), peak


def print_avg_peak_weight(peak, modes=5, depth=40, shots=100, network="brickwall"):
    plt.figure(figsize=(8, 5))
    # kernel density estimate
    sns.kdeplot(peak, fill=True, color="blue")
    plt.xlabel("Peak Probability")
    plt.ylabel("Density")
    plt.title(f"Continuous Distribution of Peak Weights ({shots} Trials)")
    legend_text = f"Modes={modes}, Depth={depth}, Network={network}"
    plt.legend([legend_text], loc="upper right", frameon=True)
    plt.show()


def decay_with_system_size(modes,  depth, shots, network):
    peaks = []
    R = range(1, modes+1)
    for mode in R:
        print("\t",mode)
        a = avg_peak_weight(mode,  depth, shots, network)
        peaks.append(a[0])
    return (R, peaks)


def print_decay_with_system_size(peaks, modes,  depth, shots, network):
    x_axis, y_axis = zip(*peaks)
    ax = plt.figure(figsize=(10, 5)).gca()
    ax.plot(x_axis, y_axis, marker='>', linestyle='-', color='b', markersize=6)

    ax.xlabel("Number of Modes")
    ax.ylabel("Average Peak Weight")
    ax.legend([f"network={network}"], loc="upper right", frameon=True, prop={'family': 'Cambria'})
    ax.grid(False)

    plt.tight_layout(pad=1)
    plt.show()


def depth_variation(modes, depth, shots, network):
    peaks = []
    R = range(1,depth+1)
    for depth in R:
        print(depth)
        peak = avg_peak_weight(modes, depth, shots, network)[0]
        peaks.append(peak)
    return (R, peaks)


def print_variation(parameters):
    ax = plt.figure(figsize=(5, 5)).gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))

    l = []
    networks = ['brickwall', 'pollman']
    for n in networks:
        x_axis, y_axis = depth_variation(*parameters, n)
        l.append(ax.scatter(x_axis, y_axis, color=next(cc), marker='o')) # should b failrly random

    ax.set_xlabel("random realizations", fontsize=10)
    ax.set_title("Distribution of peakedness", fontsize=11)
    legend = ax.legend(l, networks, loc="upper right", frameon=True)
    ax.grid(visible=False)

    plt.tight_layout(pad=1)
    plt.show()
    ax.add_artist(legend)

modes = 5
depth = 40
network="brickwall"
print_distributions((modes, depth, network))



"""

# with open('outputfile', 'w') as sys.stdout:
for seed in range(1):
    random.seed(42)
    modes=2
    target = boson_sampling(modes=modes, depth=40, network="pollman", sample=False)
    target = tf.convert_to_tensor(target, dtype=tf.float32)
    optimized_params, loss = boson_sampling_with_SGD(modes=modes, depth=10, network='pollman',
                        target=target, steps=500, learning_rate=0.1)
    print(loss)
        # print("optimized params:", optimized_params)

plt.plot(cost_progress)
plt.xlabel("Step")
plt.show()
"""