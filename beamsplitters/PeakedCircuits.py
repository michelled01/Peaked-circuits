#!/usr/bin/env python
# coding: utf-

import strawberryfields as sf
from strawberryfields.ops import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import itertools
import tensorflow as tf
import numpy as np
import random

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
    ket[(1,)*modes] = 1.0 # TODO: reduce system size (find smth convenient that works)

    with prog.context as q:
        sf.ops.Ket(ket) | q
        # for i in range(modes):
        #     Fock(1) | q[i]
        
        if network == "brickwall":
            brickwall_like(q, modes, depth)
        else:
            pollman_like(q, modes, depth)

        if (sample == True):
            MeasureFock() | q
            eng = sf.Engine("fock", backend_options={"cutoff_dim": + 1})
            results = eng.run(prog)
            samples = results.samples
            return samples
        
    results = eng.run(prog)
    state = results.state
    assert state.is_pure==True # local preparation will produce a non-pure state
    return results.state
    # TODO: update all other calls to boson_sampling


def boson_sampling_with_SGD(modes=5, depth=10, network='brickwall',
                       target_dist=None, steps=50, learning_rate=0.1):
    # 0. set up the pqc (essentailly the same thing as rqc but with optimized params)
    # 1. run the rqc
    # 2. conduct gradient descent with target: rqc, psi: pqc

    # Initial parameters
    passive_sd = 0.1
    cutoff = modes + 1

    eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})
    prog = sf.Program(modes)

    tf_params = []

    ket = np.zeros([cutoff]*modes, dtype=np.float32)
    ket[(1,)*modes] = 1.0 # m photons in m modes

    with prog.context as q:
        sf.ops.Ket(ket) | q
        
        if network == "brickwall":
            tf_params += brickwall_like(q, modes, depth, True, passive_sd)
        else:
            tf_params += pollman_like(q, modes, depth, True, passive_sd)
        
    prog.params(*[f"param_{i}" for i in range(len(tf_params))])
    
    def cost(weights):

        # TODO: imporove  this
        mapping = {f"param_{i}": weights[i] for i in range(len(weights))}

        state = eng.run(prog, args=mapping).state

        probs = state.all_fock_probs()
        
        tv = tf.reduce_sum(tf.abs(probs-target_dist))/2 # 0 <= tv distance <= 1
        
        cost = tv # tf.abs(tv - 1) # similar to |<psi|U|0> - 1|

        return cost

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for step in range(steps):
        if eng.run_progs:
            eng.reset()

        with tf.GradientTape() as tape:
            loss = cost(tf_params)
            
        gradients = tape.gradient(loss, tf_params)
        opt.apply_gradients(zip(gradients, tf_params))

        cost_progress.append(loss.numpy())
        print("Probability at step {}: {}".format(step, loss))

    optimized_parameters = [p.numpy() for p in tf_params]

    return optimized_parameters, loss


def print_distribution(probs, modes, shots, depth, network):
    # TODO: sort buckets by height (should be porter-thomas like)
    # implement SGD based off prev work add a peaking scatterplotjust like in print_variation
    x_axis = []
    y_axis = []
    for idx in np.ndindex(probs.shape):
        prob = probs[idx].item()
        y_axis.append(prob)
        x_axis.append(','.join(map(str, idx)) if prob > 0.01 else " ")
        if prob > 1e-6:
            print(f"State |{','.join(map(str, idx))}⟩: {prob:.6f}")

    plt.figure(figsize=(10, 5))
    plt.bar(x_axis, y_axis, color="blue", alpha=0.7)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Fock State")
    plt.ylabel("Probability")
    plt.title(f"Boson Sampling Output Distribution ({shots} Trials)")
    legend_text = f"Modes = {modes}, Depth={depth}, Network={network}"
    plt.legend([legend_text], loc="upper right", frameon=True)
    plt.show()


def avg_peak_weight(modes=5, depth=40, shots=100, network="brickwall"):
    peak = []
    for i in range(shots):
        probs = boson_sampling(modes, depth, network, sample=False)
        peak.append(np.max(probs))

    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak), peak


def print_avg_peak_weight(peak, modes=5, depth=40, shots=100, network="brickwall"):
    import seaborn as sns
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

    colors = ['#20786d', '#a16e1b']
    cc = itertools.cycle(colors)
    l = []
    networks = ['brickwall', 'pollman']
    for n in networks:
        x_axis, y_axis = depth_variation(*parameters, n)
        l.append(ax.scatter(x_axis, y_axis, color=next(cc), marker='o')) # should b failrly random

    ax.set_xlabel("random realizations", fontfamily='Cambria', fontsize=10)
    ax.set_title("Distribution of peakedness", fontfamily='Cambria', fontsize=11)
    legend = ax.legend(l, networks, loc="upper right", frameon=True, prop={'family': 'Cambria'})
    ax.grid(visible=False)

    plt.tight_layout(pad=1)
    plt.show()
    ax.add_artist(legend)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.sans-serif"] = ["Computer Modern Roman"]
plt.style.use("default")
cost_progress = []

# with open('outputfile', 'w') as sys.stdout:
tf.get_logger().setLevel('ERROR')
for seed in range(1):
    random.seed(42)
    modes=2
    target_dist = boson_sampling(modes=modes, depth=40, network="pollman", sample=False).all_fock_probs()
    target_dist = tf.convert_to_tensor(target_dist, dtype=tf.float32)
    optimized_params, loss = boson_sampling_with_SGD(modes=modes, depth=1, network='pollman',
                        target_dist=target_dist, steps=500, learning_rate=0.1)
    print(loss)
        # print("optimized params:", optimized_params)

plt.plot(cost_progress)
plt.xlabel("Step")
plt.show()