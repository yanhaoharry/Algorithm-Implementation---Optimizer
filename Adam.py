import numpy as np
import math

# fitting Y = X1 + 2*X2
# Loss Function: MSE
def Adam(x, y, theta, learning_rate = 0.01, iterations = 1000, threshold=0.0001, momentum = 0.1, beta1 = 0.9, beta2 = 0.9):
    m = len(y)

    # initialize
    error = 0
    mt = theta1
    vt = theta1
    e = 0.00000001

    # define a for loop
    for i in range(iterations):
        j = np.random.randint(8)
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # stop iteration
        if abs(error) <= threshold:
            break

        gradient = x[j] * (np.dot(x[j], theta) - y[j])
        mt = beta1 * mt + (1 - beta1) * gradient
        vt = beta2 * vt + (1 - beta2) * (gradient ** 2)
        mtt = mt / (1 - (beta1 ** (i + 1)))
        vtt = vt / (1 - (beta2 ** (i + 1)))
        vtt_sqrt = np.array([math.sqrt(vtt[0]), math.sqrt(vtt[1])])  # sqrt func only works for scalar
        theta = theta - learning_rate * mtt / (vtt_sqrt + e)

    print('multi features：', 'numbers of iteration：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    X1 = np.array([[1, 1], [1, 2], [2, 2], [3, 1], [1, 3], [2, 4], [2, 3], [3, 3]])
    Y1 = np.array([3, 5, 6, 5, 7, 10, 8, 9])
#    theta1 = np.array([0.0,0.0])
    theta1 = np.zeros(2)
    Adam(X1, Y1, theta1)