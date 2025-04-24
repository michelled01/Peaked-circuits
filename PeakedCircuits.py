#!/usr/bin/env python
# coding: utf-
"""
@author: michelled01
4/28/25
"""
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from scipy.stats import entropy
import sys
import tensorflow as tf
import os


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
            Interferometer(self.params) | q
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
            eng = sf.Engine("fock", backend_options={"cutoff_dim": 2})
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
            self.collision_probs.append(np.sum(probs ** 2))
            self.shannon_entropies.append(entropy(probs, base=2))
        self.save_fig()

    def save_fig(self, overlay_data=None):
        output_dir = 'beamsplitters/results/figs'
        os.makedirs(output_dir, exist_ok=True)

        _, axs = plt.subplots(1, 1, figsize=(5, 5))
        oax = axs
        global cc3

        if overlay_data:
            oax.xaxis.set_major_locator(MultipleLocator(1))
            oax.set_xlabel("Modes")
            oax.set_ylabel("Output Weight")
            for (idx, y_vals) in overlay_data:
                color=next(cc3)
                x_vals = range(1, len(y_vals) + 1)
                oax.plot(x_vals, y_vals, linestyle='--', marker="^", color=color, label=f"$\\delta={{{idx}}}$")
            oax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Cp and entropy{RandomCircuit.fig_ctr}.png'))
        plt.yscale("log")
        plt.show()
        RandomCircuit.fig_ctr += 1
        print(RandomCircuit.fig_ctr)

    ### Main Sampling Experiments ###

    def boson_sampling(self, sample=False, record_gate_seq=False):
        prog = sf.Program(self.modes)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 2})

        with prog.context as q:
            sf.ops.Ket(self.ket) | q
            self.apply_layer(q, record_gate_seq)
            if (sample == True):
                MeasureFock() | q
                eng = sf.Engine("fock", backend_options={"cutoff_dim": 2})
                results = eng.run(prog, run_options={"shots": 5000})
                samples = results.samples
                return samples
        results = eng.run(prog)
        state = results.state
        if record_gate_seq and max(self.get_probs(state)) > self.theshold:
            self.dump_output(prog, state)
            self.replay()
        return state

    def boson_sampling_with_SGD(self, target=None, T=None, L=1, record_gate_seq=True):
        fid_progress = []
        passive_sd = 0.1
        layers = L
        t = T

        eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": 2})
        prog = sf.Program(self.modes)

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

        assert tf_params.shape == weights.shape, f"Shape mismatch: tf_params.shape = {tf_params.shape}, weights.shape = {weights.shape}" # (layers, 2*modes): modes gates w/ 2*modes parameters

        with prog.context as q:
            sf.ops.Ket(self.ket) | q
            # if optimizing the amplitude we need the original circuit
            if self.cft == 2: 
                self.apply_layer(q)
            # trainable peaking layer
            for l in range(layers):
                for i in range(t):
                    BSgate(tf_params[l][2*i], tf_params[l][2*i+1]) | (q[0], q[i+1])

        # State learning adapted from https://strawberryfields.ai/photonics/demos/run_state_learner.html
        def cost1(weights):
            mapping = {p.name: w for p, w in zip(tf_params.flatten(), tf.reshape(weights, [-1]))}
            state = eng.run(prog, args=mapping).state
            ket = state.ket()
            fidelity = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target)) ** 2
            cost = tf.abs(tf.reduce_sum(tf.math.conj(ket) * target) - 1)
            return cost, fidelity, ket

        def cost2(weights):
            mapping = {p.name: w for p, w in zip(tf_params.flatten(), tf.reshape(weights, [-1]))}
            state = eng.run(prog, args=mapping).state
            ket = state.ket()
            basis = (1,) + (0,)*(self.modes-1)
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
        eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 2})

        ket = np.zeros([self.modes] * self.modes, dtype=np.float32)
        ket[(1,) + (0,)*(self.modes-1)] = 1.0
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
                    else:
                        self._apply_gate(q, gates[l][2*i], gates[l][2*i+1], 0, i+1, record_gate_seq=True)
        state = eng.run(final_prog).state
        if record_gate_seq and max(self.get_probs(state)) > self.theshold:
            self.dump_output(final_prog, state)
            self.replay()
        eng.reset()
        return state

