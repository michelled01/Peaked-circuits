s):
#     probs = {}
#     for i in range(modes):
#         basis = tuple(1 if j == i else 0 for j in range(modes))
#         probs[basis] = state.ket()[*basis]
#     return probs

# def show_probs(state, modes):
#     probs = get_probs(state, modes)
#     for basis, state in probs.items():
#         print(basis, state)

# modes = 2
# prog = sf.Program(modes)
# cutoff = modes
# eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff})
    
# ket = np.zeros([cutoff]*modes, dtype=np.float32)
# ket[(0,)*(modes-1) + (1,)] = 1.0 # 1 photon in the first mode so input state is (1,0,...0)

# with prog.context as q:
#     sf.ops.Ket(ket )| q
#     BSgate(np.pi/4, np.pi) | (q[0], q[1])

# results = eng.run(prog)
# state = results.state
# print(state)
# a = state.all_fock_probs().flatten()
# print(a)#show_probs(state, modes)
# print(state.ket())
