import numpy as np
from scipy import optimize
inputs = np.load("./data/X.npy")[:50000]
outputs = np.load("./data/y.npy").T[:50000]
samples = inputs.shape[0]
input_size = 7
hidden_size = 30
epsilon_init = 0.12
rand_weight1 = np.random.rand(hidden_size,input_size+1) *(2 * epsilon_init) - epsilon_init
rand_weight2 =  np.random.rand(1,hidden_size+1) *(2 * epsilon_init) - epsilon_init

initial_theta = np.append(rand_weight1.reshape(rand_weight1.size),rand_weight2.reshape(rand_weight2.size))

def sigmoid(z, derivative=False):
    sigm = 1. / (1. + np.exp(-z))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
def matmul(X,Y):
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]

def cost_function(theta,X,y,m):
    Theta1 = theta[:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    Theta2 = theta[hidden_size*(input_size+1):].reshape(1,hidden_size+1)
    X=np.c_[np.ones((m,1)),X]
    a2 = sigmoid(X@Theta1.T)
    a2=np.c_[np.ones((m,1)),a2]
    h = sigmoid(a2@Theta2.T)
  
    A= -y.T*np.log(h.T)
    B = -(1-y).T*np.log(1-h).T
    A[np.isnan(A)] = 0
    B[np.isnan(B)]= 0
    J = np.sum(A)+np.sum(B)

    J/=m
    return J
 
   
def cost_function2(theta,X,y,m):
    Theta1 = theta[:hidden_size*input_size].reshape(hidden_size,input_size+1)
    Theta2 = theta[hidden_size*input_size:].reshape(1,hidden_size+1)
    X=np.c_[np.ones((m,1)),X]
    a2 = sigmoid(X@Theta1.T)
    a2=np.c_[np.ones((m,1)),a2]
    h = sigmoid(a2@Theta2.T)
    J = np.sum((h-y))**2
   
    return J

def gradients(theta,X,y,m):
    Theta1 = theta[:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    Theta2 = theta[hidden_size*(input_size+1):].reshape(1,hidden_size+1)
    Theta2_grad = np.zeros(Theta2.shape)
    Theta1_grad = np.zeros(Theta1.shape)
    X=np.c_[np.ones((m,1)),X]
    z2 = X@Theta1.T
    a2 = sigmoid(z2)
    a2=np.c_[np.ones((m,1)),a2]
    h = sigmoid(a2@Theta2.T)
    d3 = h.T-y.T
    d3 = d3.T
    Theta2_grad += d3.T@a2
    z2 = np.c_[np.ones((m,1)),z2]
    d2 = (d3@Theta2) * sigmoid(z2,derivative=True)
    Theta1_grad += (d2.T@X)[1:]
    Theta2_grad/=m
    Theta1_grad/=m
    grad = np.append(Theta1_grad.reshape(Theta1_grad.size),Theta2_grad.reshape(Theta2_grad.size))
    return grad

def num_gradients(theta,X,y,m):
    numgrad = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e =0.0001
  
    for i in range(len(theta)):
        perturb[i] = e
        loss1 = cost_function(theta - perturb,X,y,m)
        loss2 = cost_function(theta + perturb,X,y,m)
        v= (loss2 - loss1) / (2*e)
        numgrad[i] = v
        perturb[i] = 0

    return numgrad

def test(theta):
    i = np.load("./data/X_test.npy")[:100]
    o = np.load("./data/y_test.npy").T[:100]
    Theta1 = theta[:hidden_size*(input_size+1)].reshape(hidden_size,input_size+1)
    Theta2 = theta[hidden_size*(input_size+1):].reshape(1,hidden_size+1)
    X=np.c_[np.ones((len(o),1)),i]
    a2 = sigmoid(X@Theta1.T)
    a2=np.c_[np.ones((len(o),1)),a2]
    h = sigmoid(a2@Theta2.T)
    count= 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    for i in range(len(o)):
        if np.round(h[i]) == 1.0:
            if o[i] == 0:
                fp+=1
            else:
                tp+=1
        else:
            if o[i] == 1:
                fn+=1
        if o[i] == np.round(h[i]):
            count+=1

    return (count/len(o),tp/(tp+fp),tp/(tp+fn))


args = (inputs,outputs,samples)
print(cost_function(initial_theta,inputs,outputs,samples))
print(test(initial_theta))
print(1-(np.sum(outputs)/samples))
print(inputs)
#print(num_gradients(initial_theta,inputs,outputs,samples),gradients(initial_theta,inputs,outputs,samples))
ans = optimize.fmin_cg(cost_function,initial_theta,fprime=gradients,args=args)
print(np.round(ans))
print(test(ans))
