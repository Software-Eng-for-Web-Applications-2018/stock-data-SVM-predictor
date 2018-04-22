# SVM Regression
#----------------------------------
#
# This function shows how to use TensorFlow to
# solve support vector regression. We are going
# to find the line that has the maximum margin
# which INCLUDES as many points as possible
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Pedal Width

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import sys
from sklearn.preprocessing import MinMaxScaler

from DatabaseORM import session, StockPriceMinute
from DataArrayTools import ShitftAmount,TrimArray
from SupportPredictorFunctions import GetStockDataList, SaveModelAndQuit


ops.reset_default_graph()


def TrainSVMLinearRegression(session,DatabaseTables,stocksym,RelativeTimeShift,DEBUG):
    # Create graph
    sess = tf.Session()


    #Xdata = GetStockDataList(session,StockPriceMinute,'AMD');
    Xdata = GetStockDataList(session,DatabaseTables,stocksym);

    #print(Xdata)

    # Shitf the training dat by X timeuits into the "future"
    Ydata = ShitftAmount(Xdata,RelativeTimeShift)

    #Make the data arrays the same length 
    Xdata = TrimArray(Xdata,(-1*RelativeTimeShift))

    LengthOfDataSet = len(Xdata)

    train_start = 0
    train_end = int(np.floor(0.8*LengthOfDataSet))
    test_start = train_end + 1
    test_end = LengthOfDataSet

    Xdata_train = Xdata[np.arange(train_start, train_end), :]
    Ydata_train = Ydata[np.arange(train_start, train_end), :]

    Xdata_test = Xdata[np.arange(test_start, test_end), :]
    Ydata_test = Ydata[np.arange(test_start, test_end), :]

    #Scale the data -- Really more for comparison between this and other prediction outputs
    Xscaler = MinMaxScaler(feature_range=(-1, 1))
    Xscaler.fit(Xdata_train)
    Xdata_train = Xscaler.transform(Xdata_train)
    Xdata_test = Xscaler.transform(Xdata_test)

    Yscaler = MinMaxScaler(feature_range=(-1, 1))
    Yscaler.fit(Ydata_train)
    Ydata_train = Yscaler.transform(Ydata_train)
    Ydata_test = Yscaler.transform(Ydata_test)

    # This svm is only 1 dim at the moment
    Xdata_train = np.array([x[0] for x in Xdata_train])
    Xdata_test = np.array([x[0] for x in Xdata_test])

    # XdataTrainTest = [Xdata_train, Xdata_test]
    #Roll seems to be rolling several axes for some damn reason. 
    Ydata_train = np.array([y[2] for y in Ydata_train])
    Ydata_test = np.array([y[2] for y in Ydata_test])

    # YdataTrainTest = [Ydata_train, Ydata_test]

    if(DEBUG == 1):
        print("Xdata\n");
        print(Xdata)
        print("\n");
        print("Ydata")
        print(Ydata)
        print("\n")

    batch_size = 50

    # NumElementsPerRow = X_train.shape[1]
    # NumElementsOut = y_train.shape[0]

    # if(DEBUG == 1):
    #     print("Length of y_train "+ str( len( y_train) ))
    #     print("NumElementsPerRow " + str(NumElementsPerRow) + " NumElementsOut " + str(NumElementsOut))

    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None,1], dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

    # Declare loss function
    # = max(0, abs(target - predicted) + epsilon)
    # 1/2 margin width parameter = epsilon
    epsilon = tf.constant([0.5])
    # Margin term in loss
    loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))

    # Declare optimizer
    my_opt = tf.train.GradientDescentOptimizer(0.075)
    train_step = my_opt.minimize(loss)

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    epoch = 300

    # Training loop
    train_loss = []
    test_loss = []
    for i in range(epoch):
        rand_index = np.random.choice(len(Xdata_train), size=batch_size)
        rand_x = np.transpose([Xdata_train[rand_index]])
        rand_y = np.transpose([Ydata_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        
        temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([Xdata_train]), y_target: np.transpose([Ydata_train])})
        train_loss.append(temp_train_loss)
        
        temp_test_loss = sess.run(loss, feed_dict={x_data: np.transpose([Xdata_test]), y_target: np.transpose([Ydata_test])})
        test_loss.append(temp_test_loss)

        if(DEBUG == 1):
            if (i+1)%50==0:
                print('-----------')
                print('Generation: ' + str(i+1))
                print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
                print('Train Loss = ' + str(temp_train_loss))
                print('Test Loss = ' + str(temp_test_loss))


    if(DEBUG == 1):
        # Extract Coefficients
        [[slope]] = sess.run(A)
        [[y_intercept]] = sess.run(b)
        [width] = sess.run(epsilon)

        # Get best fit line
        best_fit = []
        best_fit_upper = []
        best_fit_lower = []
        for i in Xdata_train:
          best_fit.append(slope*i+y_intercept)
          best_fit_upper.append(slope*i+y_intercept+width)
          best_fit_lower.append(slope*i+y_intercept-width)


        # Plot fit with data
        plt.plot(Xdata_train, Ydata_train, 'o', label='Data Points')
        plt.plot(Xdata_train, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
        plt.plot(Xdata_train, best_fit_upper, 'r--', linewidth=2)
        plt.plot(Xdata_train, best_fit_lower, 'r--', linewidth=2)
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.title('Current Price vs Future Price')
        plt.xlabel('Current Price')
        plt.ylabel('Future Price')
        plt.show()

        # Plot loss over time
        plt.plot(train_loss, 'k-', label='Train Set Loss')
        plt.plot(test_loss, 'r--', label='Test Set Loss')
        plt.title('L2 Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('L2 Loss')
        plt.legend(loc='upper right')
        plt.show()

    ModelName = 'SVM'+ stocksym
    SaveModelAndQuit(sess,ModelName)


TrainSVMLinearRegression(session,StockPriceMinute,'AMD',1,1)