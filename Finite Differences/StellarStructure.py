import numpy as np

data = np.array([ 5.90, 7.12, 8.20 ])[::-1]
h = 0.1

def der(index, h):
    return (3.0 * data[index] - 4.0 * data[index - 1] + data[index - 2]) / (2.0 * h)


print(f"Ex 2: {der(2, 0.1)}")
