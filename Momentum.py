import numpy as np


# fitting Y = X1 + 2*X2
# Loss Function: MSE
def Momentum(x, y, theta, learning_rate = 0.01, iterations = 1000, threshold=0.0001, momentum = 0.1):
    m = len(y)

    # initialize
    error = 0
    gradient = 0 # initialize gradient

    # define a for loop
    for i in range(iterations):
        j = np.random.randint(8)
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # stop iteration
        if abs(error) <= threshold:
            break

        gradient = momentum * gradient + learning_rate * (x[j] * (np.dot(x[j], theta) - y[j]))
        theta -= gradient

    print('multi features：', 'numbers of iteration：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    X1 = np.array([[1, 1], [1, 2], [2, 2], [3, 1], [1, 3], [2, 4], [2, 3], [3, 3]])
    Y1 = np.array([3, 5, 6, 5, 7, 10, 8, 9])
#    theta1 = np.array([0.0,0.0])
    theta1 = np.zeros(2)
    Momentum(X1, Y1, theta1)
