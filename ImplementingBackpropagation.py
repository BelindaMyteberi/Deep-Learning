#The following code implements the forward pass of this network in numpy. Here, you are asked to implement the backward pass,
#and obtain the gradient with respect to the weight and bias parameters.
import numpy, utils
# 1. Get the data and parameters
X,T = utils.getdata()
W,B = utils.getparams()
A = [X]
# 2. Run the forward pass
for i in range(3):
     A.append(numpy.maximum(0,A[-1].dot(W[i])+B[i]))
Y = A[-1].dot(W[3])+B[3]

# 3. Compute the error
err = ((Y-T)**2).mean()

# 4. Error backpropagation (TODO: replace by your code)

import solution
DW,DB = solution.exercise1(W,B,A,Y,T)

# 5. Show error gradient w.r.t. the 1st weight parameter
print(numpy.linalg.norm(DW[0][0,0]))