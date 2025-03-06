import matplotlib.pyplot as plt

def draw_fock_circuit(bs_gates, num_modes):
    """
    Visualizes a Fock-space photonic circuit with BSgates.

    Parameters:
    - bs_gates: List of tuples (theta, phi, mode1, mode2)
    - num_modes: Number of optical modes
    """
    fig, ax = plt.subplots(figsize=(8, num_modes))

    # Draw horizontal mode lines
    y_positions = {mode: num_modes - mode for mode in range(num_modes)}  # Flip order for visualization
    for mode in range(num_modes):
        ax.plot([0, len(bs_gates) + 1], [y_positions[mode]] * 2, 'k', lw=2)

    # Draw beamsplitters
    for i, (theta, phi, mode1, mode2) in enumerate(bs_gates):
        y1, y2 = y_positions[mode1], y_positions[mode2]
        x = i + 1  # Place beamsplitters in order
        
        # Draw vertical line for beamsplitter
        ax.plot([x, x], [y1, y2], 'b', lw=2, label="BS" if i == 0 else "")

        # Label the beamsplitter with θ, φ
        ax.text(x + 0.2, (y1 + y2) / 2, f"({theta:.2f}, {phi:.2f})", fontsize=10, color="blue")

    # Formatting
    ax.set_xticks([])
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Mode {m}" for m in range(num_modes)])
    ax.set_ylim(-1, num_modes + 1)
    ax.set_xlim(0, len(bs_gates) + 1)
    ax.set_title("Fock-Space Beamsplitter Circuit", fontsize=14)
    ax.invert_yaxis()  # Flip so Mode 0 is on top

    plt.show()


# Example usage
bs_gates = [
    (1.685, 1.983, 0, 1),
    (0.787, 4.119, 2, 3),
    (5.464, 5.597, 1, 2),
    (2.821, 2.886, 3, 4),
    (0.4836, 4.215, 0, 1),
    (5.16, 1.537, 2, 3),
    (5.511, 1.528, 1, 2),
    (3.459, 1.209, 3, 4),
    (0.6401, 5.725, 0, 1),
    (5.155, 4.775, 2, 3),
    (2.752, 2.151, 1, 2),
    (3.434, 0.574, 3, 4),
    (4.754, 3.66, 0, 1)
]

draw_fock_circuit(bs_gates, num_modes=5)