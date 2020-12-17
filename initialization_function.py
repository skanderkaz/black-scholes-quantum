import numpy as np
from utils import get_output

from copy import deepcopy
# Class used to approximate the initial ansatz using SGD




class initializer:
    def __init__(self, ansatz_operator, K, a, T, s_min, s_max):
        
        self._ansatz_operator = ansatz_operator
        self._num_qubits = ansatz_operator._num_qubits        
        self._backend = ansatz_operator._backend
        self._thetas = ansatz_operator._thetas
        self._depth = ansatz_operator._depth
        self._shots = ansatz_operator._shots
                
        self._true_res, self._gamma0 = self.initial_state(self._num_qubits, \
                                                          K, a, T, s_min, s_max)
        
        self.hist_errors = []
    
    @staticmethod
    def initial_state(n, K, a, T, s_min, s_max):
        def v_price(x, K, a, T):
            return np.exp(-a*x)*np.maximum(np.exp(x)-K,0)
        x_min = np.log(s_min)
        x_max = np.log(s_max)
        X = np.linspace(x_min, x_max, 2**n)
        psi0 = np.vectorize(lambda x: v_price(x, K, a, T))(X)
        gamma0 = 1/np.sqrt(np.sum(psi0**2))
        psi0 = psi0*gamma0
        return psi0, gamma0
        
    def modifiedCir(self, delta, theta_ind):
        theta_mod = np.array(self._thetas)[:]
        theta_mod[theta_ind] = theta_mod[theta_ind] + delta
        ansatz_mod = deepcopy(self._ansatz_operator)
        ansatz_mod.construct_circuit(theta_mod, self._depth)
        
        ansatz_mod_cir = ansatz_mod._ansatz_circuit
        res = get_output(ansatz_mod_cir, self._backend, self._num_qubits, self._shots)
        err = np.linalg.norm(res-self._true_res)
        #err1 = np.sum(np.abs(res1-self._true_res))
        return err

    def getGrad(self, ds, ind):
        res1 = self.modifiedCir(ds, ind)
        res2 = self.modifiedCir(-ds, ind)
        return (res1-res2)/(2*ds)
    
    def getGradVect(self, ds):
        n_thetas = len(self._thetas)
        vect = np.vectorize(lambda ind:self.getGrad(ds, ind))
        return vect(range(n_thetas))
    
    def gradDescent(self, alpha, epsilon, ds):
        self.hist_errors = []
        i=0
        

        while True:
            """gradVect = self.getGradVect(ds)
            new_thetas = self._thetas[:] - alpha*gradVect
            self._ansatz_operator.construct_circuit(new_thetas, self._depth)
            self._thetas = new_thetas"""
            
            for j in range(len(self._thetas)):
                grad_j = self.getGrad(ds, j)
                new_thetas = self._thetas[:]
                new_thetas[j] = self._thetas[j] - alpha*grad_j
                self._ansatz_operator.construct_circuit(new_thetas, self._depth)
                self._thetas = new_thetas
            

            
            output = get_output(self._ansatz_operator._ansatz_circuit,\
                                self._backend, self._num_qubits, self._shots)
            #error = np.sum(np.abs(output-self._true_res))
            error = np.linalg.norm(output-self._true_res)
            self.hist_errors.append(error)
           
            i+=1
            if i%10==0:
                print("Iteration {} of gradient descent, error={}".\
                      format(i, error))
            if np.linalg.norm(error) <=epsilon:
                return self._ansatz_operator, error, False
            if i>=50000:
                return self._ansatz_operator, error, True
            
    def initialize(self, init_depth, max_depth, alpha, eps, ds):

        depth = init_depth
        while depth <= max_depth:
            print('Depth='+str(depth))
            self._ansatz_operator, error, timeout = self.gradDescent(alpha,\
                                                            eps, ds)
            print("Error for depth={} is {}".format(depth, error))
            
            if timeout:
                depth+=1
                self._ansatz_operator.reset_new_depth(depth)
                print('error={}, depth increased by 1'.format(error))
            else:
                print('Solution is found, error={}'.format(error))
                return self._ansatz_operator
        raise ValueError("The algorithm did not converge and a different ansatz is needed") 