import numpy as np
from utils import get_output
from qiskit import Aer, QuantumCircuit
from ansatz import Ansatz

class AnsatzOperator(Ansatz):
    
    def __init__(self, num_qubits, matrix,\
                 backend="statevector_simulator",\
                 shots=1024, depth=2):
        
        super().__init__(num_qubits, matrix, \
                 backend="statevector_simulator",\
                 shots=1024, depth=2)
            
        self._num_qubits = num_qubits
        self._depth = depth
        self._thetas = np.random.rand(num_qubits+(2*num_qubits-1)*depth)
        self._num_thetas = len(self._thetas)
        self._backend = Aer.get_backend(backend)
        self._shots = shots
        self.construct_circuit(self._thetas, self._depth)
    
    def construct_circuit(self, thetas, depth):
        n = self._num_qubits
        self._thetas = thetas
        self._depth = depth
        
        qc = QuantumCircuit(n)
        qc.x(0)
        k=0
        for i in range(n-1):
            qc.h(i+1)
        for i in range(n):
            qc.ry(self._thetas[k],i)
            k+=1
        for _ in range(depth):
            for i in range(n-1):
                qc.cry(self._thetas[k],i,i+1)
                k+=1
            for i in range(n):
                qc.ry(self._thetas[k],i)
                k+=1     
        self._ansatz_circuit = qc
    
        
    def measure_real_part(self, qc):
        
        res = get_output(qc, self._backend, self._num_qubits, self._shots)
        result_0 = np.sum([np.abs(x)**2 for i,x in enumerate(res) if i%2==0])
        
        return 2*result_0 - 1 
        
    def measure_aij(self, i, j):
        
        n = self._num_qubits
        qc = QuantumCircuit(n+1)
        qc.h(0)
        qc.x(1)
        qc.h(range(2,n+1))
        k=0
        
        if i>j:
            i,j=j,i
        elif i==j:
            return qc, 0.
        
        ind=i
        
        def add_meas(l,k):
            nonlocal ind, qc, j;
            if (k+1==ind):
                qc.x(0)
                qc.cy(0,l)
                if ind==j:
                    return True
                else:
                    ind=j
            return False
        
        
        
        for k in range(n+ self._depth*(2*n-1)):
            ctrl= False
            if k-n<0:
                l= k+1
            elif (k-n)>=0 and (k-n)%(2*n-1)<n-1:
                l = (k-n)%(2*n-1)+2
                ctrl=True
            else:
                l = (k-n)%(2*n-1)-n+2
            
            ended = add_meas(l,k)
            if not ctrl:
                qc.ry(self._thetas[k],l)
            else:
                qc.cry(self._thetas[k], l-1, l)
            k+=1
            if ended:
                qc.h(0)
                result =  0.25*self.measure_real_part(qc)
                return qc, result
            
        return qc, 0
    
    
    def measure_ci(self, h_elements, lambdas, i):

        n = self._num_qubits
        total = 0
        
        
        for j, h_j in enumerate(h_elements):
            ang_lamb = np.angle(lambdas[j])
            norm_lamb = np.abs(lambdas[j])
            
            qc = QuantumCircuit(n+1)
            qc.h(0)
            qc.u1(ang_lamb+(np.pi/2), 0)
            qc.x(1)
            qc.h(range(2,n+1))
            
            
            def add_meas(l,k):
                nonlocal qc,i;
                if ((k+1)==i):
                    qc.x(0)
                    qc.cy(0,l)
                    return True
                return False
            
            k=0
            measured = False
            for k in range(n+ self._depth*(2*n-1)):
                ctrl= False
                
                if k-n<0:
                    l = k+1
                elif (k-n)>=0 and (k-n)%(2*n-1)<n-1:
                    l = (k-n)%(2*n-1)+2
                    ctrl=True
                else:
                    l = (k-n)%(2*n-1)-n+2
                
                if not measured:
                    measured = add_meas(l,k)
                
                if not ctrl:
                    qc.ry(self._thetas[k],l)
                else:
                    qc.cry(self._thetas[k], l-1, l)
                k+=1
            
            qc.x(0)
            
            for m, gate_ind in enumerate(h_j):
                
                if gate_ind==0:
                    continue
                elif gate_ind==1:
                    qc.cx(0,m+1)
                elif gate_ind==2:
                    qc.cy(0,m+1)
                else:
                    qc.cz(0,m+1)
            
            qc.h(0)
            result = self.measure_real_part(qc)
            total += 0.5*norm_lamb*result
        
        return qc, total.real