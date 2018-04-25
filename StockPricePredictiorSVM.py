# SVM Regression

#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#

#Based upon examples from the tensorflow cookbook

import os
import argparse;
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import sys
from sklearn.preprocessing import MinMaxScaler

from DatabaseORM import session, StockPriceMinute, StockPriceDay
from DataArrayTools import ShitftAmount,TrimArray
from SupportPredictorFunctions import GetStockDataList, SaveModelAndQuit


ops.reset_default_graph()
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
tf.app.flags.DEFINE_string('sym', '', 'Stock Symbol')  
tf.app.flags.DEFINE_integer('shiftamount', 1, 'Amount of time we wish to attept to predict into the future')
tf.app.flags.DEFINE_integer('DEBUG', 0, 'Enable the debugging output') 
tf.app.flags.DEFINE_integer('RT', 0, 'Future 0 or Historical 1') 
FLAGS = tf.app.flags.FLAGS


def TrainSVMLinearRegression(session,DatabaseTables,stocksym,RelativeTimeShift,DEBUG,typeString):
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

    if(DEBUG == 1):
        print("Xdata\n");
        print(Xdata)
        print("\n");
        print("Ydata_train")
        print(Ydata)
        print("\n")

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

    #XdataTrainTest = [Xdata_train, Xdata_test]
    #Roll seems to be rolling several axes for some damn reason. 
    Ydata_train = np.array([y[0] for y in Ydata_train])
    Ydata_test = np.array([y[0] for y in Ydata_test])

    # YdataTrainTest = [Ydata_train, Ydata_test]

    if(DEBUG == 1):
        print("Xdata_train\n");
        print(Xdata_train)
        print("\n");
        print("Ydata_train")
        print(Ydata_train)
        print("\n")

    batch_size = 50

    #NumElementsPerRow = Xdata_train.shape[1]
    # NumElementsOut = y_train.shape[0]

    # if(DEBUG == 1):
    #     print("Length of y_train "+ str( len( y_train) ))
    #     print("NumElementsPerRow " + str(NumElementsPerRow) + " NumElementsOut " + str(NumElementsOut))

    # Initialize placeholders
    X = tf.placeholder(shape=[None,1], dtype=tf.float32)
    Y = tf.placeholder(shape=[None,1], dtype=tf.float32)

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    # Declare model operations
    Out = tf.add(tf.matmul(X, A), b)

    # Declare loss function
    # = max(0, abs(target - predicted) + epsilon)
    # 1/2 margin width parameter = epsilon
    epsilon = tf.constant([0.5])
    # Margin term in loss
    loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(Out, Y)), epsilon)))

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
        #rand_x = Xdata_train[rand_index]
        #rand_y = Ydata_train[rand_index]
        sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})
        
        temp_train_loss = sess.run(loss, feed_dict={X: np.transpose([Xdata_train]), Y: np.transpose([Ydata_train])})
        train_loss.append(temp_train_loss)
        
        temp_test_loss = sess.run(loss, feed_dict={X: np.transpose([Xdata_test]), Y: np.transpose([Ydata_test])})
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
    #SaveModelAndQuit(sess,ModelName)

     # Export model
    export_path_base = FLAGS.work_dir + 'SVM_'+ typeString + '_'+stocksym
    export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
    #export_path = ModelName + '/' + export_path 
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(Out) #THIS IS IMPORTANT!!! NOT THE PLACEHOLDER!!!!!!!!

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'input': tensor_info_x},
          outputs={'output': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          'prediction':
              prediction_signature,
      },
      legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')
    sys.exit(0)


def main():

    #user input
    # learning rate 
    parser = argparse.ArgumentParser();
    parser.add_argument('--sym', dest= 'sym', default='');
    parser.add_argument('--DEBUG',type=int, dest= 'debug', default=0);
    parser.add_argument('--shiftamount',type=int, dest= 'shiftamount', default=1);
    parser.add_argument('--RT',type=int, dest= 'history', default=0);

    args = parser.parse_args();

    print("Input Arguments: ") 
    print(args)
    if(args.history == 0):
        TrainSVMLinearRegression(session,StockPriceMinute,args.sym,args.shiftamount,args.debug,'RT');
    else:
        TrainSVMLinearRegression(session,StockPriceDay,args.sym,args.shiftamount,args.debug,'PAST');

main();
