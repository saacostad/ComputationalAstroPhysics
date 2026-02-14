import  numpy as np 


F = np.array([0.0035, 0.0055, 0.0080])
h = 2.0


def first_OD(index, dx):
    return (F[index + 1] - F[index - 1]) / dx

def second_OD(index, dx):
    return ( F[index+1] - 2.0*F[index] + F[index-1]) / (dx**2)


print(f"Task A: {first_OD(1, h)}")
print(f"Task B: {second_OD(1, h)}")


