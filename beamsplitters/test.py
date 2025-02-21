import strawberryfields as sf
from strawberryfields.ops import Dgate
import matplotlib.pyplot as plt
import numpy as np

# Initialize engine and program
eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": 20})
prog = sf.Program(1)

alpha = 1.0  # Displacement parameter (coherent amplitude)

with prog.context as q:
    Dgate(alpha) | q[0]

# Run the program
print("here")
result = eng.run(prog)
print("here!")
state = result.state
print("here?")
# Plot Wigner function (phase space distribution)
x = np.linspace(-5, 5, 100)
p = np.linspace(-5, 5, 100)
X, P = np.meshgrid(x, p)
wigner = state.wigner(0, X, P)

plt.figure(figsize=(6, 6))
plt.contourf(X, P, wigner, levels=100, cmap="viridis")
plt.colorbar(label="Wigner Function")
plt.xlabel("x (position quadrature)")
plt.ylabel("p (momentum quadrature)")
plt.title(f"Wigner Function of Coherent State (|\u03b1|={alpha})")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
