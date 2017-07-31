import tensorflow as tf
import matplotlib.pyplot as plt #we will use it to draw the learning curve after training the network

train_x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
train_y = [[0], [1], [1], [0]]

INPUT_NEURONS = 2
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 1

NUM_OF_EPOCHS = 100000

"""(tf.float32, [None, 2]) specifies the datatype and the dimensions of the data.
#Since we don't know the number of the training data, we make it None which means it accepts any from the user.
#2 specifies that we have 2 input bits
"""
x = tf.placeholder(tf.float32, [None, 2])
y_target = tf.placeholder(tf.float32, [None, 1])

"""
1- Create the Input-to-hidden weights and bias matrices from the given figure. 
They should be Variable datatype because they will be changed during the learning process
"""

# Write your code here
hidden_layer_weights = tf.Variable(tf.float32,[[-0.99,1.05,0.19],[-0.43,-0.44,-0.30]])
hidden_layer_bias = tf.Variable(tf.float32,[1,1,1])

"""
2- Get the values of the hidden layer by multiplying the features with the weight matrix [Input to Hidden feedforward]
Apply the hidden layer activation to the multiplication result
"""

# Write your code here
#getting theta transpose . X
hidden_layer_mulitply = tf.matmul(x,hidden_layer_weights)
#adding the bias
hidden_layer_mulitply = hidden_layer_mulitply + hidden_layer_bias
#using the sigmoid function as the activation function

active_hidden_layer = tf.nn.sigmoid(hidden_layer_mulitply)

"""
3- Create the hidden-to-output weights and bias matrices from the given figure. 
They should be Variable datatype because they will be changed during the learning process
"""

# Write your code here

hidden_to_output_weights = tf.Variable(tf.float32,[[0.18],[1.11],[-0.26]])
hidden_to_output_bias = tf.Variable(tf.float32,[1])


"""
4- Get the values of the output layer by multiplying the hidden layer with the weight matrix [Hidden to Output feedforward]
Apply the output layer activation to the multiplication result
"""

# Write your code here
hiddem_to_output_multiply = tf.matmul(active_hidden_layer,hidden_to_output_weights) + hidden_layer_bias

# activate
active_hidden_to_output = tf.nn.sigmoid(hiddem_to_output_multiply)


mean_squared_error = 0.5 * tf.reduce_sum((tf.square(active_hidden_to_output - y_target)))
train = tf.train.GradientDescentOptimizer(0.1).minimize(mean_squared_error)


"""
Initiate a Tensorflow graph and session variables
"""
session = tf.Session()
session.run(tf.initialize_all_variables())

errors = []
epochs = []

for i in range(0, NUM_OF_EPOCHS):
    session.run(train, feed_dict={x: train_x, y_target: train_y})

    if i % 10 == 0:
        print("Iteration number: ", i, "\n")
        error = session.run(mean_squared_error, feed_dict={x: train_x, y_target: train_y})
        print("Cost: ", error, "\n")
        errors.append(error)
        epochs.append(i)

        if error < 0.01:
            plt.title("Learning Curve using mean squared error cost function")
            print("Cost: ", error, "\n")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Cost")
            plt.plot(epochs, errors)
            plt.show()

            break