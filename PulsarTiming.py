import numpy as np 


time = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
vel = np.array([200.0, 205.2, 210.8, 216.9, 223.5])
h = 0.5 

def F_der(index, h):
    return (vel[index + 1] - vel[index]) / h 


def B_der(index, h):     
    return (vel[index] - vel[index-1]) / h 


def C_der(index, h):
    return ( vel[index + 1] - vel[index - 1] ) / (2.0 * h)



print(f"Task A: {F_der(0, h)}")
print(f"Task B: {B_der(4, h)}")
print(f"Task C: {C_der(2, h)}")
