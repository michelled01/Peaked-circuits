#!/usr/bin/env python
# coding: utf-

import io
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import PchipInterpolator
import seaborn as sns
import strawberryfields as sf
from strawberryfields.ops import *
from strawberryfields.utils import operation
from scipy.stats import entropy
import sys
import tensorflow as tf
import os

np.random.seed(42)
tf.random.set_seed(42)
cc1 = itertools.cycle(sns.color_palette("rocket",10))
cc2 = itertools.cycle(sns.color_palette("mako", 10))
cc3 = itertools.cycle(sns.cubehelix_palette(8))
plt.rcParams["font.family"] = "Cambria"


class RandomCircuit:
    fig_ctr = 0
    
    def __init__(self, modes, depth, network, cost_function_type=1, steps=50, learning_rate=0.1, theshold=0.7):
        self.modes = modes
        self.depth = depth
        self.network = network
        self.params = self.init_params()
        self.cft = cost_function_type 
        self.steps = steps
        self.learning_rate = learning_rate
        self.theshold = theshold
        self.ket = np.zeros([self.modes] * self.modes, dtype=np.float32)
        self.ket[(1,) + (0,)*(modes-1)] = 1.0 # 1 photon in first mode
        self.gate_seq = []
        self.collision_probs = []
        self.shannon_entropies = []

    ### Layer Initializations ###

    def init_params(self):
        if self.network == 'haar-random':
            # https://pennylane.ai/qml/demos/tutorial_haar_measure
            A, B = np.random.normal(size=(self.modes, self.modes)), np.random.normal(size=(self.modes, self.modes))
            Z = A + 1j * B
            Q, R = np.linalg.qr(Z)
            Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(self.modes)])
            return np.dot(Q, Lambda)
        else:
            return np.random.uniform(0, 2 * np.pi, (self.depth, self.modes))

    def apply_layer(self, q, record_gate_seq):
        if self.network == 'haar-random':
            Interferometer(self.params), q
        elif self.network == 'brickwall':
            self._apply_brickwall(q, record_gate_seq)
        elif self.network == 'pollman':
            self._apply_pollman(q, record_gate_seq)
        else:
            Rgate(0), q[0]
        return q

    def _apply_brickwall(self, q, record_gate_seq):
        for i in range(self.depth):
            for j in range(i%2!=0, self.modes-1, 2):
                self._apply_gate(q, self.params[i, j], self.params[i, j+1], j, j+1, False, record_gate_seq)
                # BSgate(self.params[i, j], self.params[i, j+1]) | (q[j], q[j+1])

    def _apply_pollman(self, q, record_gate_seq):
        for i in range(self.depth):
            if i % 2 == 0:
                for j in range(self.modes - 1):
                    self._apply_gate(q, self.params[i, j], self.params[i, j+1], j, j+1, False, record_gate_seq)
                    # BSgate(self.params[i, j], self.params[i, j+1]) | (q[j], q[j+1])
            # else:
            #     for j in range(self.modes):
            #         self._apply_gate(Rgate(self.params[i, j]), q, j)

    def _apply_gate(self, q, theta, phi, i, j, inverse, record_gate_seq):
        if inverse == True:
            BSgate(theta, phi).H | (q[i], q[j])
        else:
            BSgate(theta, phi) | (q[i], q[j])
        if record_gate_seq:
            self.gate_seq.append((theta, phi, i, j, inverse))

    ### Utilities ###

    def get_probs(self, state):
        probs = []
        for i in range(self.modes):
            basis = tuple(1 if j == i else 0 for j in range(self.modes))
            probs.append(state.fock_prob(basis))
        return np.array(probs)

    def show_probs(self, state):
        print(self.get_probs(state))

    def CP(self, probs):
        return np.sum(probs ** 2)

    def dump_output(self, prog, state):
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.show_probs(state)
        prog.print()
        sys.stdout = old_stdout
        lines = []
        for line in buffer.getvalue().split("\n"):
            if line.strip() == "" or "[0. 0. 0. 0. 0. 0.]" in line or "[1. 0. 0. 0. 0. 0.]" in line:
                continue
            lines.append(line)
        filtered_output = "\n".join(lines)
        with open("outfile.txt", "w") as f:
            f.write(filtered_output)

    def replay(self):
        self.collision_probs = []
        self.shannon_entropies = []
        for t in range(-1, len(self.gate_seq)):
            prog = sf.Program(self.modes)
            eng = sf.Engine("fock", backend_options={"cutoff_dim": self.modes})
            with prog.context as q:
                sf.ops.Ket(self.ket) | q
                for (theta, phi, i, j, inv) in self.gate_seq[:t+1]:
                    if inv:
                        BSgate(theta, phi).H | (q[i], q[j])
                    else:
                        BSgate(theta, phi) | (q[i], q[j])
            state = eng.run(prog).state
            eng.reset()
            probs = self.get_probs(state)
            # self.show_probs(state)
            self.collision_probs.append(np.sum(probs ** 2))
            self.shannon_entropies.append(entropy(probs, base=2))
        self.save_fig()

    def save_fig(self, overlay_data=None):
        output_dir = 'beamsplitters/results/figs'
        os.makedirs(output_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        cax, eax = axs
    
        cax.xaxis.set_major_locator(MultipleLocator(10))
        cax.set_xlabel("Circuit Depth")
        cax.set_ylabel("Collision Probability")
        if overlay_data:
            for (idx, y_vals, cy_error) in overlay_data["collision_probs"]:
                color=next(cc3)
                x_vals = range(1, len(y_vals) + 1)
                cax.plot(x_vals, y_vals, linestyle='-', color=color, label=f"$\\delta={{{idx}}}$")
                (_, caps, _) = cax.errorbar(x_vals, y_vals, yerr=cy_error, color=color, fmt='^', markersize=8, capsize=20)
                for cap in caps: cap.set_markeredgewidth(.3)
        else:
            cax.plot(range(1, len(self.collision_probs) + 1), self.collision_probs, marker='o', label="Collision Probability")
        cax.legend()
   
        eax.xaxis.set_major_locator(MultipleLocator(10))
        eax.set_xlabel("Circuit Depth")
        eax.set_ylabel("Shannon Entropy")
        if overlay_data:
            for (idx, y_vals, ey_error) in overlay_data["shannon_entropies"]:
                color=next(cc3)
                x_vals = range(1, len(y_vals) + 1)
                eax.plot(x_vals, y_vals, linestyle='-', color=color, label=f"$\\delta={{{idx}}}$")
                (_, caps, _) = eax.errorbar(x_vals, y_vals, yerr=ey_error, color=color, fmt='^', markersize=8, capsize=20)
                for cap in caps: cap.set_markeredgewidth(.3)

                # eax.plot(range(1, len(data) + 1), data, linestyle='-', color=color, label=f"$\\delta={{{idx}}}$")
        else:
            eax.plot(range(1, len(self.collision_probs) + 1), self.collision_probs, marker='o', label="Shannon Entropy")
        eax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Cp and entropy{RandomCircuit.fig_ctr}.png'))
        plt.show()
        RandomCircuit.fig_ctr += 1
        print(RandomCircuit.fig_ctr)

    ### Main Sampling Experiments ###

    def boson_sampling(self, sample=False, record_gate_seq=False):
        prog = sf.Program(self.modes)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.modes})

        with prog.context as q:
            sf.ops.Ket(self.ket) | q
            self.apply_layer(q, record_gate_seq)
            if (sample == True):
                MeasureFock() | q
                eng = sf.Engine("fock", backend_options={"cutoff_dim": self.modes})
                results = eng.run(prog, run_options={"shots": 5000})
                samples = results.samples
                return samples
        results = eng.run(prog)
        state = results.state
        # if record_gate_seq and max(self.get_probs(state)) > self.theshold:
        #     self.dump_output(prog, state)
        #     self.replay()
        return state

    def boson_sampling_with_SGD(self, target=None, T=None, record_gate_seq=True):
        fid_progress = []
        passive_sd = 0.1
        layers = 1 # technically not depth because layers arent parallelized
        t = max(T[0]*(self.modes-1)//T[1],1)

        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.modes})
        prog = sf.Program(modes)

        bs_theta = tf.random.normal(shape=[layers], stddev=passive_sd) 
        bs_phi = tf.random.normal(shape=[layers], stddev=passive_sd)
        weights = tf.convert_to_tensor([bs_theta, bs_phi]*t)
        weights = tf.Variable(tf.transpose(weights))
        
        tf_params = []
        names = ["bs{}".format(i) for i in range(2*t)]
        for i in range(layers):
            tf_params_names = ["{}_{}".format(n, i) for n in names]
            tf_params.append(prog.params(*tf_params_names))
        tf_params = np.array(tf_params)

        assert tf_params.shape == weights.shape, f"Shape mismatch: tf_params.shape = {tf_params.shape}, weights.shape = {weights.shape}" # (layers, 2*modes) <-- modes gates so 2*modes paramters

        with prog.context as q:
            sf.ops.Ket(self.ket) | q
            # if optimizing the amplitude we need the original circuit
            if self.cft == 2: 
                self.apply_layer(q)
            # trainable peaking layer
            for l in range(layers):
                for i in range(t):
                    BSgate(tf_params[l][2*i], tf_params[l][2*i+1]) | (q[0], q[i+1])

        # from tensorflow import keras
        # kl = keras.losses.KLDivergence()
        
        def cost1(weights):
            mapping = {p.name: w for p, w in zip(tf_params.flatten(), tf.reshape(weights, [-1]))}
            state = eng.run(prog, args=mapping).state
            ket = state.ket()
            # show_probs(modes, state)
            fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target)) ** 2
            cost = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target) - 1)
            return cost, fidelity, ket

        def cost2(weights):
            mapping = {p.name: w for p, w in zip(tf_params.flatten(), tf.reshape(weights, [-1]))}
            state = eng.run(prog, args=mapping).state
            ket = state.ket()
            # show_probs(modes, state)
            basis = (1,) + (0,)*(modes-1)
            probability = tf.abs(state.fock_prob(basis))
            cost = abs(probability - 1)
            return cost, probability, ket

        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        for step in range(self.steps):
            if eng.run_progs:
                eng.reset()
            with tf.GradientTape() as tape:
                if self.cft == 1:
                    loss, fid, ket = cost1(weights)
                else:
                    loss, fid, ket = cost2(weights)
            fid_progress.append(fid.numpy())
            gradients = tape.gradient(loss, weights)
            opt.apply_gradients(zip([gradients], [weights]))

            print("Rep: {} Cost: {:.4f} Fidelity: {:.4f}".format(step, loss, fid))

        optimized_parameters = np.array([p.numpy() for p in weights]).tolist()
        return self.run_circuit(t, optimized_parameters, record_gate_seq)


    def run_circuit(self, t, opt_params, record_gate_seq):
        layers, _ = len(opt_params), len(opt_params[0])

        final_prog = sf.Program(self.modes)
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": self.modes})

        ket = np.zeros([self.modes] * self.modes, dtype=np.float32)
        ket[(1,) + (0,)*(modes-1)] = 1.0
        gates = []
        if self.cft == 1:
            for p in reversed(opt_params):
                lst = p[::-1]
                for i in range(0, len(lst)-1, 2):
                    lst[i], lst[i + 1] = lst[i + 1], lst[i]
                gates.append(lst)
        else:
            gates = opt_params

        with final_prog.context as q:
            sf.ops.Ket(self.ket) | q
            self.apply_layer(q, record_gate_seq)
            for l in range(layers):
                for i in range(t):
                    if self.cft == 1:
                        self._apply_gate(q, gates[l][2*i], gates[l][2*i+1], 0, t-i, inverse=True, record_gate_seq=True)
                        # BSgate(gates[l][2*i], gates[l][2*i+1]).H | (q[0], q[t-i])
                    else:
                        self._apply_gate(q, gates[l][2*i], gates[l][2*i+1], 0, i+1, record_gate_seq=True)
                        # BSgate(gates[l][2*i], gates[l][2*i+1]) | (q[0], q[i+1])
        state = eng.run(final_prog).state
        if record_gate_seq and max(self.get_probs(state)) > self.theshold:
            self.dump_output(final_prog, state)
            self.replay()
        eng.reset()
        return state


def postselect(parameters, trials):
    modes, depth, network = parameters[:3]
    
    overlay_data = {"collision_probs": [], "shannon_entropies": []}

    for t in np.linspace(0, 0.5, 3):
        avg_probs = {}
        min_cps = [np.ones(modes) for _ in range(depth)]
        max_cps = [np.zeros(modes) for _ in range(depth)]
        min_entropies = [np.ones(modes) for _ in range(depth)]
        max_entropies = [np.zeros(modes) for _ in range(depth)]
        for d in range(depth):
            ctr = 0
            print("depth",d)
            min_cp = 1
            max_cp = 0
            min_entropy = modes
            max_entropy = 0
            while ctr < trials:
                circuit = RandomCircuit(modes, d, network, *parameters[3:])
                state = circuit.boson_sampling(record_gate_seq=False)
                probs = circuit.get_probs(state)
                if max(probs) > t:
                    min_cp = min(min_cp, np.sum(probs ** 2))
                    max_cp = max(max_cp, np.sum(probs ** 2))
                    min_entropy = min(min_entropy, entropy(probs, base=2))
                    max_entropy = max(min_entropy, entropy(probs, base=2))
                    if d not in avg_probs:
                        avg_probs[d] = probs
                    else:
                        avg_probs[d] += probs
                    ctr += 1
            min_cps[d] = min_cp
            max_cps[d] = max_cp
            min_entropies[d] = min_entropy
            max_entropies[d] = max_entropy

        for k, prob in avg_probs.items():
            avg_probs[k] = prob / trials
            assert 0 <= all(avg_probs[k]) <= 1

        collision_probs = [np.sum(avg_probs[d] ** 2) for d in range(depth)]
        shannon_entropies = [entropy(avg_probs[d], base=2) for d in range(depth)]

        if t==0: print("entropies",shannon_entropies) 
        if t==0: print("min_entropies",min_entropies) 
        if t==0: print("max_entropies",max_entropies) 
        cy_error = [max(abs(collision_probs[d]-min_cps[d]), abs(min_cps[d] - collision_probs[d])) for d in range(depth)]
        ey_error = [max(abs(shannon_entropies[d]-min_entropies[d]), abs(max_entropies[d] - shannon_entropies[d])) for d in range(depth)]

        overlay_data['collision_probs'].append((t,collision_probs, cy_error))
        overlay_data['shannon_entropies'].append((t,shannon_entropies, ey_error))
        if t==0:
            print("cy_error", cy_error)
            print("ey_error", ey_error)
    circuit = RandomCircuit(*parameters)
    circuit.save_fig(overlay_data)
            # can consider many things here,
            # what proportion of random circuits are peaked (should be rare)
            # criteria for change in circuit structure:
            # if identify stark contrast in circuit structure at time t, measure operator norm between two different parts to see if its a trivial inverse


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

def print_distributions(parameters, n, cft, steps):
    ax = plt.figure(figsize=(5, 5)).gca()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    l = []
    circuit = RandomCircuit(*parameters, cft, steps)
    rqc = circuit.boson_sampling()
    r_axes = order(rqc.all_fock_probs())
    idn = {'random layers': r_axes}
    for i in range(1,n+1):
        pqc = circuit.boson_sampling_with_SGD(rqc.ket(), (i,n))
        p_axes = order(pqc.all_fock_probs())
        idn[f"random + $\\frac{{{i}}}{{{n}}}m$ peaking layers"] = p_axes
    for label, data in idn.items():
        x, y = data
        pchip = PchipInterpolator(x, y) # nonnegative cubic interpolator
        x_smooth = np.linspace(min(x), max(x), 100)
        y_smooth = pchip(x_smooth)
        color=next(cc1 if cft==1 else cc2)
        ax.scatter(x, y, marker="^", color=color)
        l.append(ax.plot(x_smooth, y_smooth, color=color, label=label)[0])
        plt.ylim(0, 1)
        # l.append(ax.plot(*data, color=next(cc1), marker="^", label=label)[0])

    ax.set_xlabel("single-photon modes", fontsize=10)
    str = "state learning" if cft==1 else "amplitude maximization"
    network = parameters[2]
    ax.set_title(f"Output weight {str} ({network})", fontsize=11)
    legend = ax.legend(l, idn.keys(), loc="upper right", frameon=True)
    
    ax.grid(visible=False)
    plt.tight_layout(pad=1)
    plt.show()
    ax.add_artist(legend)


def avg_peak_weight(parameters, shots=100):
    peak = []
    for i in range(shots):
        circuit = RandomCircuit(*parameters)
        probs = circuit.boson_sampling()
        peak.append(np.max(probs))

    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak), peak

def print_avg_peak_weight(peak, modes, depth, network, shots=100):
    plt.figure(figsize=(8, 5))
    # kernel density estimate
    sns.kdeplot(peak, fill=True, color="blue")
    plt.xlabel("Peak Probability")
    plt.ylabel("Density")
    plt.title(f"Continuous Distribution of Peak Weights ({shots} Trials)")
    legend_text = f"Modes={modes}, Depth={depth}, Network={network}"
    plt.legend([legend_text], loc="upper right", frameon=True)
    plt.show()

def decay_with_system_size(parameters, shots):
    peaks = []
    R = range(1, modes+1)
    for mode in R:
        a = avg_peak_weight(*parameters, shots)
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

def depth_variation(parameters, shots):
    peaks = []
    R = range(1,depth+1)
    for depth in R:
        peak = avg_peak_weight(*parameters, shots)[0]
        peaks.append(peak)
    return (R, peaks)

def print_variation(parameters):
    ax = plt.figure(figsize=(5, 5)).gca()
    ax.xaxis.set_major_locator(MultipleLocator(5))

    l = []
    networks = ['brickwall', 'pollman']
    for n in networks:
        x_axis, y_axis = depth_variation(*parameters, n)
        l.append(ax.scatter(x_axis, y_axis, color=next(cc1), marker='o')) # should b failrly random

    ax.set_xlabel("random realizations", fontsize=10)
    ax.set_title("Distribution of peakedness", fontsize=11)
    legend = ax.legend(l, networks, loc="upper right", frameon=True)
    ax.grid(visible=False)

    plt.tight_layout(pad=1)
    plt.show()
    ax.add_artist(legend)

modes = 5
depth = 20
network = 'brickwall'
n = 8
cft = 1
steps = 100
theshold = 0
trials = 50
record_gate_seq = True # set to True if running boson_sampling graph ONLY
parameters = (modes, depth, network, cft, steps, 0.1, theshold)
parameter = (modes, depth, network)
# print_distributions(parameter, n, cft, steps)
postselect(parameters,trials)
# circuit.boson_sampling_with_SGD(target=a, T=(8,8), record_gate_seq=True)
# circuit = RandomCircuit(modes=modes, depth=depth, network=network, cost_function_type=cft, steps=steps, theshold=theshold)
# target = circuit.boson_sampling(record_gate_seq=False).ket()
# state = circuit.boson_sampling_with_SGD(target, (n,n), record_gate_seq=True)
# if max(circuit.get_probs(state)) > theshold:
#     circuit.replay()

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