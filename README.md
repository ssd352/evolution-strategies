
# Assignment 1 Part B. Using Evolution Strategies for Non-differentiable Loss Function

Notebook adapted from [https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb](https://github.com/udacity/CarND-LeNet-Lab/blob/master/LeNet-Lab-Solution.ipynb)  
You can find the paper on evolution strategies at:
[https://arxiv.org/abs/1703.03864](https://arxiv.org/abs/1703.03864)


## Load Data

First we load the MNIST data, which comes pre-loaded with TensorFlow.


```
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False, one_hot=True)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))
```

    Image Shape: (28, 28, 1)
    
    Training Set:   55000 samples
    Validation Set: 5000 samples
    Test Set:       10000 samples



```
import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))
```

    Updated Image Shape: (32, 32, 1)


## Setup TensorFlow
Here, we define some of the hyperparameters of the network.

Also, the training happens really fast with the following hyperparameters. As a result, 2 epochs would suffice to achieve an accuracy of around 90% but I set `EPOCHS` to 15 just to be on the safe side (because of the huge amount of stochasticity in this problem). Feel free to reduce it if you want.



```
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
num_samples = 200
EPOCHS = 50
BATCH_SIZE = 128
acc_sigma = 4.0  # 1.75
rate = 5.0 / 55000
```

## SOLUTION: Implement LeNet-5
Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.

### Input
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.

### Architecture
**Layer 1: Convolutional.** The output shape should be 28x28x6.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 14x14x6.

**Layer 2: Convolutional.** The output shape should be 10x10x16.

**Activation.** Your choice of activation function.

**Pooling.** The output shape should be 5x5x16.

**Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.

**Layer 3: Fully Connected.** This should have 120 outputs.

**Activation.** Your choice of activation function.

**Layer 4: Fully Connected.** This should have 84 outputs.

**Activation.** Your choice of activation function.

**Layer 5: Fully Connected (Logits).** This should have 10 outputs.

### Output
Return the result of the logits layer.


```
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6
    conv1_W = tf.get_variable("conv1_W", shape=(5, 5, 1, 6),
           initializer=tf.contrib.layers.xavier_initializer())
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.get_variable("conv2_W", shape=(5, 5, 6, 16),
           initializer=tf.contrib.layers.xavier_initializer())
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
#     fc1_b = tf.Variable(tf.zeros(120))
#     fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    fc1 = tf.layers.dense(fc0, 120, name="fc1", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    
    # SOLUTION: Activation.
#     fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
#     fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#     fc2_b  = tf.Variable(tf.zeros(84))
#     fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    fc2 = tf.layers.dense(fc1, 84, name="fc2", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         activation=tf.nn.relu)
    
    # SOLUTION: Activation.
#     fc2    = tf.nn.relu(fc2)

    logits = tf.layers.dense(fc2, 10, name="logits", 
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return logits
```

## Features and Labels
Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.


```
x = tf.placeholder(tf.float32, (None, 32, 32, 1), name="input_data")
one_hot_y = tf.placeholder(tf.int32, (None, 10), name='true_labels')
# epsi = tf.placeholder(tf.float32, shape=(None, 10))  
# one_hot_y = tf.one_hot(y, 10)
```

## Training Pipeline
Create a training pipeline that uses the model to classify MNIST data.


```
logits = LeNet(x)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
```

## Estimating the gradient using ES algorithm

At first, I used `tf.while_loop` for gradient estimation but eventually I found a shorter (and hopefully more efficient) way to do it.
$$F=1-Accuracy$$
$$ \nabla F(l) \approx \frac{1}{n\sigma}\sum_{i=1}^{n}{F(l + \sigma\epsilon_i)\epsilon_i} $$
where $n$ is `num_samples`, $\sigma$ is `acc_sigma` and $l$ is `logits`.


```
# cnt = tf.constant(0)
# acc_mean = tf.Variable(tf.zeros([BATCH_SIZE, 10]), name="acc_mean", trainable=False, validate_shape=False)
# indices = tf.where(tf.equal(tf.argmax(logits + acc_sigma * epsi, 1), tf.argmax(one_hot_y, 1)))
# acc_mean = tf.reduce_sum(1-tf.gather_nd(epsi, indices=indices) / acc_sigma / num_samples, axis=0)

log_repl = tf.expand_dims(logits, 2)
y_repl = tf.expand_dims(one_hot_y, 2)
# log_repl = tf.reshape(tf.tile(logits, [num_samples]), [1, 1, num_samples)
# y_repl = tf.reshape(tf.tile(one_hot_y, num_samples), [1, 1, num_samples])
epsi = tf.random_normal([tf.shape(one_hot_y)[0], tf.shape(one_hot_y)[1], num_samples])

F = tf.cast(tf.not_equal(tf.argmax(log_repl + acc_sigma * epsi, 1), tf.argmax(y_repl, 1)), tf.float32)
acc_mean = tf.reduce_mean(tf.expand_dims(F, 1) * epsi, axis=2) / acc_sigma

# # acc_sum2 = tf.Variable(0.0, name="acc_sum2", trainable=False)
# acc_sum2 = tf.zeros(10)
# acc_mean2 = tf.zeros(10)
# def cond(i, assign):
#     return i < tf.shape(x)[0]
#     return True
# def body(i, assign):
#   global acc_mean
#   epsi = tf.random_normal(tf.shape(logits))
#   indices = tf.where(tf.not_equal(tf.argmax(logits[i, :] + acc_sigma * epsi, 1), tf.argmax(one_hot_y[i, :], 1)))
#   assign = tf.assign(acc_mean[i, :], tf.reduce_sum(tf.gather_nd(epsi, indices=indices) / acc_sigma / num_samples, axis=0), validate_shape=False)
#   i += 1
#   return i, assign
# # #     global acc_sum2, acc_mean2
#     F = -tf.cast(tf.equal(my_argmax[i],
#                                     tf.argmax(one_hot_y, 1)), tf.float32)
#     sum_so_far += F * epsi[i, :] / acc_sigma / num_samples
#     sum_so_far.set_shape([10])
# #     # acc_mean2 += (F * epsi / acc_sigma - acc_mean2) / (i + 1)
#     i += 1
# #     i += tf.shape(F)[0]
# #     sum_so_far /= tf.cast(i, tf.float32)
#     return i, sum_so_far
# cnt, acc_mean = tf.while_loop(cond, body, [tf.constant(0), tf.zeros([10])])
# cnt, assign =  tf.while_loop(cond, body, [tf.constant(0), tf.constant(0)])
```

## Using the Estimate in the Training Pipeline
### First Attempt : RegisterGradient
My first attempt was to define gradients for the non-differentiable operations.


```
# Define new operation and gradient
# tf.Graph.create_op("NegAcc")
# https://uoguelph-mlrg.github.io/tensorflow_gradients/

# @tf.RegisterGradient("NegAccGrad")
# def neg_acc_grad(op, grad):
    
#     print(tmp.shape)
#     print(op.inputs[0].shape, tf.constant(1.0).shape)
#     return tf.expand_dims(tf.constant(1.0), 0), None  # tf.constant(0.0)

# @tf.RegisterGradient("AccSumGrad")
# def acc_sum_grad(op, grad):
#   return acc_sum
# @tf.RegisterGradient("ArgMaxGrad")
# def arg_max_grad(op, grad):
#     tmp = acc_mean
#     print(tmp.shape)
#     return tmp, None

# with tf.get_default_graph().gradient_override_map({'Equal': 'NegAccGrad',
#                                                    'Cast': 'Identity', "ArgMax": "ArgMaxGrad"}):
#     loss_operation = tf.cast(tf.equal(tf.argmax(logits + acc_sigma * epsi, 1),
#                                        tf.argmax(one_hot_y, 1)), tf.float32)
# print(loss_operation.shape)
# with tf.get_default_graph().gradient_override_map({'acc_sum': 'AccSumGrad'}):
#   loss_operation = acc_mean + tf.stop_gradient(tf.cast(tf.equal(tf.argmax(logits, 1),
#                                        tf.argmax(one_hot_y, 1)), tf.float32) - acc_mean)
```

### Second Attempt: Using Taylor Series to Approximate F
I also tried using the following loss (inspired by taylor series) to approximate the non-differentiable loss function. This was a successful attempt and has been used in this notebook.
$$Loss = \nabla F(l) l \Longrightarrow \nabla_l Loss = \nabla F(l)$$


```
# F = -tf.cast(tf.equal(tf.argmax(logits, 1),
#                                     tf.argmax(one_hot_y, 1)), tf.float32)
# loss_operation = tf.stop_gradient(F) + tf.matmul(logits, tf.expand_dims(tf.stop_gradient(acc_mean), 1), name="taylor_series")
# loss_operation = tf.matmul(logits, tf.stop_gradient(tf.expand_dims(acc_mean, 1)), name="taylor_series")
prod = tf.multiply(tf.stop_gradient(acc_mean), logits)
loss_operation = tf.reduce_sum(prod, axis=1)
```

### Third Attempt: Gradients Multiplication

Another one of my attempts was to compute the gradients of the logits and then multiply the very last one by the estimated gradient.



```
# training_operation = optimizer.minimize(loss_operation, global_step=global_step)
# grads_and_vars = optimizer.compute_gradients(logits, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

# grads = [grad for grad in tf.gradients(logits, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) if grad is not None]
# last_grad = tf.get_default_graph().get_tensor_by_name(name="gradients_2/add_4_grad/Reshape_1:0")
# last_grad = grads[-1]
# last_grad = acc_mean + last_grad

# last_grad = grads[-2]
# last_grad = tf.get_default_graph().get_tensor_by_name(name="gradients_2/MatMul_2_grad/MatMul_1:0")
# last_grad = tf.matmul(last_grad, tf.expand_dims(acc_mean, 1))

# for grad, var in grads_and_vars:
#   if grad is not None:
#     grad *= tf.gacc_mean
# training_operation = optimizer.apply_gradients(grads_and_vars)

```

### Last Attempt: Using the Gradient Estimate to Update Logits

This attempt was in accordance with Prof. Ray's proposition and successful. 
$$logits_{new} = logits - \frac{\alpha
}{n\sigma}\sum_{i=1}^{n}{F(logits + \sigma\epsilon_i)\epsilon_i} $$
$$Loss = \big|\big|logits_{new} - logits\big|\big|_2^2$$
However, I chose the second one over this, specifically because here we are doing the logit update manually whereas in the second approach, we come up with a loss function and let the Optimizer (in this case Adam) handle the update.


```
# new_logits = logits - rate * acc_mean
# loss_operation = tf.losses.mean_squared_error(labels=tf.stop_gradient(new_logits), predictions=logits)
```

## Computing the Gradients of the Network Parameters

After defining a loss function for this problem, we compute its gradient with respect to the trainable network parameters and use it to learn them.


```
grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
training_operation = optimizer.apply_gradients(grads_and_vars)
```

## Creating Summaries


```
def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)

for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/histogram", var)

add_gradient_summaries(grads_and_vars)
merged_summary_op= tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('logs-student/')
summary_writer.add_graph(tf.get_default_graph())
```

## Model Evaluation
The function `evaluate` evaluates the accuracy of the model for a given dataset.


```
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

def evaluate(X_data, y_data):
    eval_batch_size = 128
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, eval_batch_size):
        batch_x, batch_y = X_data[offset:offset+eval_batch_size], y_data[offset:offset+eval_batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, one_hot_y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

## Train the Model
Here, we run the training data through the training pipeline to train the model.

The model is saved after training.


```
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]

            _, summaries = sess.run([training_operation, merged_summary_op],
                                      feed_dict={x: batch_x, one_hot_y: batch_y})

#             if end % 5000 == 0: 
#                 summary_writer.add_summary(summaries, global_step=offset)
#                 validation_accuracy = evaluate(X_validation, y_validation)
#                 print("{} of 55000\nValidation Accuracy = {:.3f}".format(end, validation_accuracy))

        print()
        print("EPOCH {} ...".format(i+1))
        validation_accuracy = evaluate(X_validation, y_validation)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    
    EPOCH 1 ...
    Validation Accuracy = 0.787
    
    
    EPOCH 2 ...
    Validation Accuracy = 0.831
    
    
    EPOCH 3 ...
    Validation Accuracy = 0.841
    
    
    EPOCH 4 ...
    Validation Accuracy = 0.852
    
    
    EPOCH 5 ...
    Validation Accuracy = 0.855
    
    
    EPOCH 6 ...
    Validation Accuracy = 0.858
    
    
    EPOCH 7 ...
    Validation Accuracy = 0.862
    
    
    EPOCH 8 ...
    Validation Accuracy = 0.862
    
    
    EPOCH 9 ...
    Validation Accuracy = 0.867
    
    
    EPOCH 10 ...
    Validation Accuracy = 0.870
    
    
    EPOCH 11 ...
    Validation Accuracy = 0.872
    
    
    EPOCH 12 ...
    Validation Accuracy = 0.874
    
    
    EPOCH 13 ...
    Validation Accuracy = 0.872
    
    
    EPOCH 14 ...
    Validation Accuracy = 0.878
    
    
    EPOCH 15 ...
    Validation Accuracy = 0.877
    
    
    EPOCH 16 ...
    Validation Accuracy = 0.879
    
    
    EPOCH 17 ...
    Validation Accuracy = 0.877
    
    
    EPOCH 18 ...
    Validation Accuracy = 0.881
    
    
    EPOCH 19 ...
    Validation Accuracy = 0.879
    
    
    EPOCH 20 ...
    Validation Accuracy = 0.884
    
    
    EPOCH 21 ...
    Validation Accuracy = 0.883
    
    
    EPOCH 22 ...
    Validation Accuracy = 0.881
    
    
    EPOCH 23 ...
    Validation Accuracy = 0.956
    
    
    EPOCH 24 ...
    Validation Accuracy = 0.960
    
    
    EPOCH 25 ...
    Validation Accuracy = 0.962
    
    
    EPOCH 26 ...
    Validation Accuracy = 0.965
    
    
    EPOCH 27 ...
    Validation Accuracy = 0.967
    
    
    EPOCH 28 ...
    Validation Accuracy = 0.971
    
    
    EPOCH 29 ...
    Validation Accuracy = 0.970
    
    
    EPOCH 30 ...
    Validation Accuracy = 0.972
    
    
    EPOCH 31 ...
    Validation Accuracy = 0.975
    
    
    EPOCH 32 ...
    Validation Accuracy = 0.974
    
    
    EPOCH 33 ...
    Validation Accuracy = 0.974
    
    
    EPOCH 34 ...
    Validation Accuracy = 0.976
    
    
    EPOCH 35 ...
    Validation Accuracy = 0.975
    
    
    EPOCH 36 ...
    Validation Accuracy = 0.975
    
    
    EPOCH 37 ...
    Validation Accuracy = 0.974
    
    
    EPOCH 38 ...
    Validation Accuracy = 0.977
    
    
    EPOCH 39 ...
    Validation Accuracy = 0.978
    
    
    EPOCH 40 ...
    Validation Accuracy = 0.980
    
    
    EPOCH 41 ...
    Validation Accuracy = 0.978
    
    
    EPOCH 42 ...
    Validation Accuracy = 0.978
    
    
    EPOCH 43 ...
    Validation Accuracy = 0.978
    
    
    EPOCH 44 ...
    Validation Accuracy = 0.979
    
    
    EPOCH 45 ...
    Validation Accuracy = 0.980
    
    
    EPOCH 46 ...
    Validation Accuracy = 0.979
    
    
    EPOCH 47 ...
    Validation Accuracy = 0.979
    
    
    EPOCH 48 ...
    Validation Accuracy = 0.981
    
    
    EPOCH 49 ...
    Validation Accuracy = 0.980
    
    
    EPOCH 50 ...
    Validation Accuracy = 0.980
    
    Model saved


## Evaluate the Model
Here, we evaluate the performance of the model on the test set.


```
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    INFO:tensorflow:Restoring parameters from ./lenet
    Test Accuracy = 0.978

