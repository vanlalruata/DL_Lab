"""
Practical 1: McCulloch-Pitts Neuron from Scratch
Objective: Implement an M-P neuron class in NumPy to simulate fundamental logic
gates (AND, OR, NOT, and NOR). Show via truth tables why a single-layer M-P
thresholding fails to solve the non-linearly separable XOR gate.
"""

import numpy as np


class McCullochPittsNeuron:
    """A fixed-weight, thresholded McCulloch-Pitts neuron.
    output = 1 if (sum(w_i * x_i) + bias) >= 0 else 0
    """

    def __init__(self, weights, threshold):
        self.weights = np.array(weights, dtype=float)
        self.threshold = float(threshold)

    def activate(self, x):
        x = np.array(x, dtype=float)
        return 1 if np.dot(self.weights, x) >= self.threshold else 0


def truth_table(gate_name, neuron, inputs):
    print(f"\n=== {gate_name} ===")
    print("x\ty\tout")
    rows = []
    for x in inputs:
        out = neuron.activate(x)
        print(f"{x}\t{out}")
        rows.append(out)
    return rows


def main():
    binary_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # AND: fire only when both inputs are 1
    and_gate = McCullochPittsNeuron(weights=[1, 1], threshold=2)
    truth_table("AND", and_gate, binary_inputs)

    # OR: fire when at least one input is 1
    or_gate = McCullochPittsNeuron(weights=[1, 1], threshold=1)
    truth_table("OR", or_gate, binary_inputs)

    # NOT (single input): invert
    not_gate = McCullochPittsNeuron(weights=[-1], threshold=0)
    print("\n=== NOT ===")
    for x in [(0,), (1,)]:
        print(f"{x}\t{not_gate.activate(x)}")

    # NOR: NOT(OR)
    nor_gate = McCullochPittsNeuron(weights=[-1, -1], threshold=-1)
    truth_table("NOR", nor_gate, binary_inputs)

    # XOR: attempt with a single M-P neuron
    # There is NO single set of fixed weights/threshold that realises XOR.
    print("\n=== XOR (attempted with single M-P neuron) ===")
    print("Truth table for XOR is:")
    xor_target = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    for x, y in xor_target.items():
        print(f"{x}\t{y}")

    # Demonstrate failure: show that no threshold works for all 4 inputs.
    print("\nWhy a single M-P neuron cannot separate XOR:")
    print("XOR positive class points: (0,1) and (1,0)")
    print("XOR negative class points: (0,0) and (1,1)")
    print("These are not linearly separable; any separating line mislabels a point.")
    # Sweep a few candidate weight combinations to prove impossibility.
    candidates = [([1, 1], 1), ([1, -1], 0), ([-1, 1], 0), ([1, 1], 2)]
    for w, t in candidates:
        n = McCullochPittsNeuron(w, t)
        preds = [n.activate(x) for x in binary_inputs]
        target = [xor_target[x] for x in binary_inputs]
        print(f"weights={w} threshold={t} -> preds={preds} target={target} "
              f"match={preds == target}")


if __name__ == "__main__":
    main()
