import numpy as np
import random
import math
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms import Grover, AmplificationProblem

'''
Configuration: choose an arbitrary list of number
'''
T = [4, 1, 123, 2, 39, 7, 0, 12]
N = len(T)
num_qubits = int(np.log2(N))

def create_circuit(data, best_index):
    '''
        We recreate the quantum circuit, which mark each state |x> if T[X] > T[Y], where Y is the current best index
    '''

    solutions_indices = [
        x for x in range(len(data))
        if data[x] > data[best_index]
    ]

    # If there is no solution, we have found the maximum
    # So we retrive a circuit which didn't mark anything
    if not solutions_indices:
        circ = QuantumCircuit(num_qubits)
        return circ, False
    
    # If there are some solutions, then we create a circuit which mark each state that satisfy the condition
    circ = QuantumCircuit(num_qubits)
    
    # For each solution we apply an Z gate
    for sol_index in solutions_indices:
        # Convert the decimal number in a binary number
        binary_sol = reversed(format(sol_index, f'0{num_qubits}b'))

        # X-Gate
        for i, bit in enumerate(binary_sol):
            if bit == '0':
                circ.x(i)
        
        # Z-Gate
        circ.mcp(math.pi, list(range(num_qubits - 1)), num_qubits - 1)

        # Now we have to restore the initial state
        for i, bit in enumerate(binary_sol):
            if bit == '0':
                circ.x(i)


        circ.barrier()
    
    return circ, True


def find_max(data):
    '''
        Execute the algorithm
    '''

    # Choose a random index
    best_guess_index = random.randint(0, len(data) - 1)
    print(f"Initial guess (index y): {best_guess_index}, Value T[y]: {data[best_guess_index]}")

    num_iterations = int(np.log2(len(data))) + 1
    for i in range(num_iterations):
        print(f"Iteration: {i+1}")
        print(f"Current candidate: y = {best_guess_index}, T[y] = {data[best_guess_index]}")

        # Now we can apply the Grover search algorithm
        circ, has_solution = create_circuit(data, best_guess_index)

        if not has_solution:
            print("The current candidate is the maximum")
            break

        def is_good_state(bitstring):
            index = int(bitstring, 2)
            return data[index] > data[best_guess_index]

        # Prepare the amplification problem for Grover
        problem = AmplificationProblem(circ, is_good_state=is_good_state)

        # The Sampler pick in input a quantum circuits, execute it many times, count each result and return the probability distribution
        grover = Grover(iterations=1, sampler=Sampler())

        # Run the quantum search
        result = grover.amplify(problem)

        # Make a measurement on the most likely state
        new_candidate = result.top_measurement
        new_candidate_index = int(new_candidate, 2)

        print(f"Grover found a new candidat: x0 = {new_candidate_index}, T[x0] = {data[new_candidate_index]}")
        best_guess_index = new_candidate_index
    
    print(f"\nFinal result: Index={best_guess_index}, Maximum Value={data[best_guess_index]}")
    return best_guess_index

find_max(T)