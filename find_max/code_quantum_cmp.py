import numpy as np
import random
import math
from qiskit import QuantumCircuit
from qiskit.circuit.library import IntegerComparator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import Grover, AmplificationProblem


T = [4, 1, 123, 2, 39, 7, 0, 12]
N = len(T)
num_qubits = int(np.log2(N))
num_value_qubits = 7 # -> We need to know an upper bound for the maximum of array, for this example we will simply consider 7 bit


def apply_qrom(circ, index_qubits, value_qubits, data):
    '''
        Quantum Read Only Memory: for each indicies x, T[x] is load to the memory
        This is an auto-inverse process, if applied twice the value will return the initial one
    '''
    n_index = len(index_qubits)
    n_value = len(value_qubits)

    for idx, value in enumerate(data):
        if value == 0:
            continue

        idx_bits = format(idx, f'0{n_index}b')[::-1]
        val_bits = format(value, f'0{n_value}b')[::-1]

        control_qubits = []
        for q, bit in zip(index_qubits, idx_bits):
            if bit == '1':
                control_qubits.append(q)
            else:
                circ.x(q)
                control_qubits.append(q)
        
        for v_qubit, v_bit in zip(value_qubits, val_bits):
            if v_bit == '1':
                circ.mcx(control_qubits, v_qubit)
        
        for q, bit in zip(index_qubits, idx_bits):
            if bit == '0':
                circ.x(q)
    
    circ.barrier()

def get_oracle(data, threshold, num_index_qubits, num_value_qubits, num_ancillas):
    '''
        We recreate the quantum circuit, which mark each state |x> if T[X] > threshold (tipically the best index until now)
        So we need to create a registry for the qubits:
            - q[0 : num_index_qubits] -> the indicies of |x>
            - q[num_index_qubits : num_index_qubits + num_value_qubits] -> the values of the qubits
            - q[num_index_qubits + num_value_qubits + 1] -> the flag of the comparator
            - q[num_index_qubits + num_value_qubits + 2 : num_index_qubits + num_value_qubits + 2 + num_ancillas] -> the ancillas qubits for the comparator

    '''
    total_qubits = num_index_qubits + num_value_qubits + 1 + num_ancillas
    oracle = QuantumCircuit(total_qubits)

    index_qubits = list(range(num_index_qubits))
    value_qubits = list(range(num_index_qubits, num_index_qubits + num_value_qubits))
    flag_qubit = num_index_qubits + num_value_qubits
    ancilla_qubits = list(range(flag_qubit + 1, total_qubits))

    gate = IntegerComparator(num_state_qubits = num_value_qubits, value = threshold + 1, geq = True)
    cmp_qubits = value_qubits + [flag_qubit] + ancilla_qubits

    apply_qrom(oracle, index_qubits=index_qubits, value_qubits=value_qubits, data=data)

    oracle.append(gate, cmp_qubits)

    oracle.z(flag_qubit)

    oracle.append(gate.inverse(), cmp_qubits)
    apply_qrom(oracle, index_qubits=index_qubits, value_qubits=value_qubits, data=data)

    return oracle

def build_state_preparation(num_index_qubits, num_value_qubits, num_ancillas):
    '''
        Prepare the uniform distribution of probabilities on all the indices
    '''
    total_qubits = num_index_qubits + num_value_qubits + 1 + num_ancillas
    prep = QuantumCircuit(total_qubits)
    prep.h(range(num_index_qubits)) # Apply Hadamard only to the index qubits
    return prep

def find_max(data):
    N = len(data)
    num_index_qubits = int(np.log2(N))

    dummy_cmp = IntegerComparator(num_state_qubits=num_value_qubits, value=0, geq=True)
    num_cmp_qubits = dummy_cmp.num_qubits
    num_ancillas = num_cmp_qubits - (num_value_qubits + 1)

    sampler = Sampler()
    grover = Grover(sampler=sampler)

    state_prep = build_state_preparation(num_index_qubits=num_index_qubits, num_value_qubits=num_value_qubits, num_ancillas=num_ancillas)

    best_idx = random.randint(0, N - 1)
    print(f"Initial guess y = {best_idx}, T[y] = {data[best_idx]}")

    num_iterations = max(1, int(math.pi / 4 * math.sqrt(N)))
    for it in range(num_iterations):
        
        oracle = get_oracle(data, threshold=data[best_idx], num_index_qubits=num_index_qubits, num_value_qubits=num_value_qubits, num_ancillas=num_ancillas)

        def is_good_state(bitstring):
            idx = int(bitstring, 2)
            return data[idx] > data[best_idx]

        problem = AmplificationProblem(
            oracle=oracle,
            state_preparation=state_prep,
            is_good_state=is_good_state,
            objective_qubits=range(num_index_qubits),
        )

        result = grover.amplify(problem)

        new_x_bits = result.top_measurement
        new_idx = int(new_x_bits, 2)
        print(f"For Grover Search at iteration {it}, the max: x = {new_idx}, T[x] = {data[new_idx]}")

        if data[new_idx] > data[best_idx]:
            best_idx = new_idx
        else:
            print("No improvement found in this round (keep current y).")

    print(f"\nFinal result: index = {best_idx}, max T[x] = {data[best_idx]}")
    return best_idx

find_max(T)