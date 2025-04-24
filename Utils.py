#!/usr/bin/env python
# coding: utf-
"""
@author: michelled01
4/28/25
"""
import numpy as np
from strawberryfields.ops import *

from PeakedCircuits import *


def postselect(parameters, trials):
    modes, depth, network = parameters[:3]
    overlay_data = []
    for t in np.linspace(0, 0.8, 2):
        ord_avg_peaks = np.zeros(modes)
        ctr = 0
        while ctr < trials:
            circuit = RandomCircuit(*parameters)
            state = circuit.boson_sampling(record_gate_seq=False)
            probs = circuit.get_probs(state)
            ord_peaks = np.array(sorted(probs, reverse=True))
            if max(probs) > t:
                ord_avg_peaks += ord_peaks
                ctr += 1
        ord_avg_peaks /= trials
        assert 0 <= all(ord_avg_peaks) <= 1
        overlay_data.append((t,ord_avg_peaks))
    
    circuit = RandomCircuit(*parameters)
    circuit.save_fig(overlay_data)

def order(probs):
    state_probs = []
    for idx in np.ndindex(probs.shape):
        prob = probs[idx].item()
        if prob > 0:
            state = f"|{','.join(map(str, idx))}⟩"
            state_probs.append((state, prob))

    for state, prob in state_probs: 
        print(f"State {state}: {prob:.6f}")
    state_probs.sort(key=lambda x: x[1], reverse=True)
    return range(1, len(state_probs) + 1), tuple(zip(*state_probs))[1]

def avg_peak_weight(parameters, shots=100):
    peak = []
    for i in range(shots):
        circuit = RandomCircuit(*parameters)
        probs = circuit.boson_sampling()
        peak.append(np.max(probs))
    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak), peak