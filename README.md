# Quantum Computing — Algorithms in Qiskit

This repository collects hands-on implementations of quantum algorithms as I learn them and code them with Qiskit. The goal is to have reproducible examples with references to the original papers and notes on how to run them.

## Implemented algorithms

### 1) Maximum finding (Grover-based)

- Folder: `find_max/`
- Main file: `find_max/code_quantum_cmp.py`
- Article: [My Substack Article](https://open.substack.com/pub/vittoriosironi/p/quantum-algorithms-1?r=4cebo7&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
- Short description:
  - The algorithm searches for the index of the maximum value in a list `T` of length `N = 2^n` using an oracle that marks states with values greater than the current candidate and a Grover step to amplify the probability of measuring a better element. It repeats for about O(log N) iterations, updating the candidate.
- Contract (I/O):
  - Input: a list `T` whose length is a power of two (e.g., 8, 16, 32, ...). In the code it is defined at the top; change `T` to try other cases.
  - Output: index of the maximum and a printout of the corresponding value.
- Practical notes:
  - The oracle circuit is built dynamically based on the current candidate; if there are no larger elements, the oracle does not mark any state and the algorithm stops.
  - The current version uses 1 Grover iteration per loop; this is a simple, didactic choice.

#### References

- A. Ahuja, S. Kapoor, "A Quantum Algorithm for finding the Maximum", Department of Computer Science and Engineering, 1999; https://arxiv.org/abs/quant-ph/9911082v1
