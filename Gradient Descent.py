import numpy as np

# fitting Y = 2X
# Loss Function: MSE
# np.array([0,0]) => 2 columns, 1 row

def bgd_single(x, y, theta, learning_rate = 0.01, iterations = 1000, threshold=0.0001):

    # initialize paras
    m = len(y)
    error = 0

    # define a for loop
    for i in range(iterations):

        error = 1 / m * np.dot(((theta * x) - y), ((theta * x) - y).T)
        # stop iteration
        if abs(error) <= threshold:
            break

        theta -= 2 * learning_rate / m * (np.dot((theta * x - y),x.T ))

    print('single feature：', 'numbers of iteration： %d' % (i + 1), 'theta： %f' % theta,
          'error： %f' % error)

def bgd_multi(x, y, theta, learning_rate = 0.01, iterations = 1000, threshold=0.0001):
    # fitting Y = X1 + 2*X2
    # Loss Function: MSE
    m = len(y)

    # initialize
    error = 0

    # define a for loop
    for i in range(iterations):
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # stop iteration
        if abs(error) <= threshold:
            break

        theta -= learning_rate / m * (np.dot(x.T, (np.dot(x, theta) - y)))

    print('multi features：', 'numbers of iteration：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)



if __name__ == '__main__':
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # feature
    Y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) # label
    theta = 0
    bgd_single(X, Y, theta)

    X1 = np.array([[1, 1], [1, 2], [2, 2], [3, 1], [1, 3], [2, 4], [2, 3], [3, 3]])
    Y1 = np.array([3, 5, 6, 5, 7, 10, 8, 9])
#    theta1 = np.array([0.0,0.0])
    theta1 = np.zeros(2)
    bgd_multi(X1, Y1, theta1)