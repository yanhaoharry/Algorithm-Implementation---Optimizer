import numpy as np


# fitting Y = X1 + 2*X2
# Loss Function: MSE
def SGD(x, y, theta, learning_rate = 0.01, iterations = 1000, threshold=0.0001):
    m = len(y)

    # initialize
    error = 0

    # define a for loop
    for i in range(iterations):
        j = np.random.randint(8)
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # stop iteration
        if abs(error) <= threshold:
            break

        theta -= learning_rate * (x[j] * (np.dot(x[j], theta) - y[j]))

    print('multi features：', 'numbers of iteration：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    X1 = np.array([[1, 1], [1, 2], [2, 2], [3, 1], [1, 3], [2, 4], [2, 3], [3, 3]])
    Y1 = np.array([3, 5, 6, 5, 7, 10, 8, 9])
#    theta1 = np.array([0.0,0.0])
    theta1 = np.zeros(2)
    SGD(X1, Y1, theta1)