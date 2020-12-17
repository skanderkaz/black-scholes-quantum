from  utils import decomposeHermitian, get_output
import numpy as np

from abc import ABC, abstractmethod

class Ansatz(ABC):
    def __init__(self, num_qubits, matrix,\
                 backend="statevector_simulator",\
                 shots=1024, depth=2):
        
        self._h, self._lambdas = decomposeHermitian(matrix)
        self._depth = depth
        self._backend = backend
        self._shots = shots
        self._num_qubits = num_qubits
                                
    
    @abstractmethod
    def measure_aij(self, a, b):
        pass
    
    @abstractmethod
    def measure_ci(self, h_elements, lambdas, i):
        pass
    
    @abstractmethod
    def construct_circuit(self, thetas):
        pass
    
    def compute_A(self):
        N = len(self._thetas)
        A = np.zeros((N, N))
        for i in range(N):
            for j in range(i,N):
                A[i,j] = self.measure_aij(i+1, j+1)[1]
        return A
    
    def compute_C(self):
        C = np.vectorize(lambda i:\
                         self.measure_ci(self._h, self._lambdas, i+1)[1])\
            (range(len(self._thetas)))
        return C

    def update_circuit(self, dtau):
        A = self.compute_A()
        C = self.compute_C()
        
        theta_point = np.linalg.lstsq(A,C)[0]
        new_thetas = self._thetas[:] + dtau*theta_point

        self._thetas = new_thetas
        self.construct_circuit(new_thetas, self._depth)
        
    def execute(self, sigma, T, num_steps):
        dtau = (sigma**2)*T/num_steps
        
        final_outcomes = [list(np.abs(get_output(self._ansatz_circuit,\
                                    self._backend, self._num_qubits, self._shots)))]
        
        for i in range(num_steps):
            self.update_circuit(dtau)
            output = np.abs(get_output(self._ansatz_circuit, self._backend,\
                                       self._num_qubits, self._shots))
            final_outcomes.append(output)
            if (i+1)%20==0:
                print("Simulation {} completed".format(i+1))
        
        final_result = list(map(np.abs, final_outcomes))
        final_result = np.stack(final_result, axis=0)

        return final_result