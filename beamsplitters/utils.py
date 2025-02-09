#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 05 00:13:38 2025  

@author: md45223
"""
#%config InlineBackend.figure_formats = ['svg']
import numpy as np
import quimb as qu
import quimb.tensor as qtn

class Beamsplitter:
    def beamsplitter_uni(theta):
        """
        Constructs an 2 x 2 unitary uni representing a beamsplitter.
        Parameters: theta (float): The angle parameter for R_theta (in radians).
        """    
        R_theta = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
        return R_theta

    def rand_uni():
        r = np.random.rand() * 2*np.pi
        return Beamsplitter.beamsplitter_uni(r)
    
class Phaseshifter:
    def phase_shifter_uni(theta):
        """
        Constructs an 1 x 1 unitary uni representing a phase shifter.
        Parameters: theta (float): The phase shift parameter (in radians).
        """    
        # uni = np.eye(2, dtype=np.complex128)
        
        # uni[0, 0] = np.exp(1j * theta)
            
        return np.exp(1j * theta)
    
    def rand_uni():
       r = np.random.rand() * 2*np.pi
       return Phaseshifter.phase_shifter_uni(r)
    
def rand_uni(dtype=complex):
    return Phaseshifter.rand_uni() if (np.random.rand() > 0.5) else Beamsplitter.rand_uni()

def pollmann_circuit(psi, i_start, n_apply, list_u3, depth, modes, data_type, seed_val, Qubit_ara, uni_list, rand=False):
    # add a seq of beamsplitters, then a seq of phase shifters, then a seq beamsplitters, etc...
    gate_round=None
    if modes==0: depth=1
    if modes==1: depth=1
    c_val=0
    for r in range(depth):
        if r%2 == 0:
            for i in range(i_start, i_start+modes-1, 1):
                #print("U_e", i, i + 1, n_apply)
                G = Beamsplitter.rand_uni()
                psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
                list_u3.append(f'G{n_apply}')
                n_apply+=1
                c_val+=1
        else:
            for i in range(i_start, i_start+modes, 1):
                G = Phaseshifter.rand_uni()
                psi.gate_(G, (i), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
                list_u3.append(f'G{n_apply}')
                n_apply+=1
                c_val+=1

    return n_apply, list_u3

def brickwall_circuit(psi, i_start, n_apply, list_u3, depth, modes, data_type, seed_val, Qubit_ara, uni_list, rand=True):
    # chose random beamsplitters in each layer and fill in rest with phaseshifters
    gate_round=None
    if modes==0: depth=1
    if modes==1: depth=1
    c_val=0
    for r in range(depth):
        i = i_start
        while i < modes:
            if np.random.rand() > 0.5:
                G = Beamsplitter.rand_uni()
                psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
                list_u3.append(f'G{n_apply}')
                n_apply+=1
                c_val+=1
                i += 2
            else:
                G = Phaseshifter.rand_uni()
                psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
                list_u3.append(f'G{n_apply}')
                n_apply+=1
                c_val+=1
                i += 1

    return n_apply, list_u3

def qmps_f(L=16, in_depth=2, modes=3, data_type='float64', qmps_structure="brickwall", canon="left",  n_q_mera=2, seed_init=0, internal_mera="brickwall", uni_list = None,rand = True):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('1' * L) # single-photon source at each mode
   for i in range(L):
        t = psi[i]
        indx = 'k'+str(i)
        t.modify(left_inds=[indx])

   for t in range(L):
        psi[t].modify(tags=[f"I{t}", "MPS"])


   if canon=="left":

    for i in range(0,L-modes,1):
        #print ("qubit", i+modes, modes)
        Qubit_ara=i+modes
        if qmps_structure=="brickwall":
            n_apply, list_u3=brickwall_circuit(psi, i, n_apply, list_u3, in_depth, modes,data_type,seed_val, Qubit_ara,uni_list = uni_list,rand =rand)
        elif qmps_structure=="pollmann":
            n_apply, list_u3=pollmann_circuit(psi, i, n_apply, list_u3, in_depth, modes,data_type,seed_val, Qubit_ara,uni_list= uni_list,rand =rand)

   return psi.astype_('complex128')#, list_u3

def uni_list(dic,val_iden=0.,val_dic = 0.): #create the unitary list 
    uni_list = {}
    opt_tags = list(dic.keys())
    #for i in (opt_tags):
    #    uni_list[i] = qu.identity(4,dtype='complex128')+qu.randn((4,4))*val_iden
    if dic != None:
        for j in dic:
            uni_list[j] = dic[j].reshape(4,4).T + qu.randn((4,4))*val_dic
    return list(uni_list.values())

def norm_f(psi):
    # method='qr' is the default but the gradient seems very unstable
    # 'mgs' is a manual modified gram-schmidt orthog routine
    return psi.isometrize(method='mgs',allow_no_left_inds=True)

def average_peak_weight(L=10,depth=100, shots=1024, nphotons=100):
    peak = []
    for i in range (shots):
        psi_2 = qmps_f(L, in_depth=depth, modes=L-1, qmps_structure="brickwall", canon="left",  n_q_mera=2, seed_init=10, internal_mera="brickwall")
        peak.append(nphotons * max(abs((psi_2^all).data.reshape(2**L))**2))
    return np.mean(peak), np.std(peak)/np.sqrt(shots), np.max(peak)

def negative_overlap(psi, target):
    return - abs((target.H & psi)^all) ** 2  # minus so as to minimize

print(average_peak_weight())
