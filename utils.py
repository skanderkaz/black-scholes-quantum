import numpy as np
import itertools
from functools import reduce
from qiskit import execute

def get_output(circ, backend, n, shots):   

    res = 0
    if backend.name()=='statevector_simulator':
        result = execute(circ, backend, shots=shots).result()
        res = result.get_statevector(circ)
        
    elif backend.name()=='qasm_simulator':
        qc = circ.copy()
        qc.measure_all(inplace=True)
        result = execute(qc, backend, shots=shots).result()
        counts = result.get_counts()
        temp_counts = {}

        for i in range(2**(n+1)):
            temp_counts[format(i, 'b').zfill(n+1)] = 0
        for x in temp_counts.keys():
            if x in counts.keys():
                temp_counts[x] += counts[x]
                
        res = np.array(list(temp_counts.values()))/shots

    return res


pauli_X = np.array([[0,1],[1,0]])
pauli_Y = np.array([[0,-1j],[1j,0]])
pauli_Z = np.array([[1,0],[0,-1]])
gates = [np.eye(2), pauli_X, pauli_Y, pauli_Z]
    
def decomposeHermitian(matrix):
    n = int(np.log2(matrix.shape[0]))
    k=0
    h = np.zeros((4**n, n))
    lambdas = np.zeros(4**n,dtype=complex)
    for x in itertools.product([0,1,2,3],repeat=n):
        gate = reduce(np.kron, [gates[i] for i in x[::-1]])
        lambdas[k] = (1/(2**n))*np.trace(gate.conj().T @ matrix)
        h[k,:] = x
        k+=1
    h = h[lambdas!=0,:]
    lambdas = lambdas[lambdas!=0]
    return h, lambdas


def createMatrixEuropeanCall(n, b, s_min, s_max):
    H = np.zeros((2**n,2**n))
    H[0,0] = -b
    H[-1,-1] = -b
    x_min = np.log(s_min)
    x_max = np.log(s_max)
    dx = (x_max-x_min)/(2**n-1)
    for i in range(1,(2**n)-1):
        H[i,i-1] = 0.5*(1/dx**2)
        H[i,i] = -1/(dx**2)
        H[i,i+1] = 0.5*(1/dx**2)        
    return H
