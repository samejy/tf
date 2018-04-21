import numpy as np
import random
import tensorflow as tf


def create_network(layer_sizes: list):
    layer_params = []
    data_size = layer_sizes[0]

    # create weights and bias parameters for all layers
    for cur, nxt in zip(layer_sizes[:-1],layer_sizes[1:]):
        wgts = tf.Variable(tf.random_normal([cur,nxt]))
        bias = tf.Variable(tf.random_normal([nxt]))
        layer_params.append({'weights':wgts,'bias':bias})

    # input data placeholder
    x = tf.placeholder(tf.float32, shape=(None,data_size))

    # input layer
    input = tf.add( \
                    tf.matmul(x, layer_params[0]['weights']), \
                    layer_params[0]['bias'])

    input_layer = tf.nn.relu(input)

    # list of all layers we create
    layers = []
    layers.append(input_layer)
    cur_layer = input_layer

    # subtract 2 as we are starting at second layer
    # when enumerating below
    last_layer_idx = len(layer_params) - 2

    # create the actual computation for all layers
    for i, lparams in enumerate(layer_params[1:]):
        nxt_layer = tf.add( \
                        tf.matmul(cur_layer, lparams['weights']), \
                        lparams['bias'])
        if i != last_layer_idx:
            nxt_layer = tf.nn.relu(nxt_layer)

        layers.append(nxt_layer)
        cur_layer = nxt_layer

    # the output layer:
    return (x, cur_layer)


def train_network(all_data, all_labels, n_epochs, batch_sz):
    nsamples = len(all_data)
    nlabels = len(all_labels)
    if nsamples != nlabels:
        raise Exception("data and labels are differing sizes");w


    nsamples = int(nsamples * 0.8)
    data = all_data[:nsamples]
    labels = all_labels[:nsamples]
    test_x = all_data[nsamples+1:]
    test_y = all_labels[nsamples+1:]

    sample_sz = len(data[0])
    label_sz = len(labels[0])

    x, model = create_network([sample_sz, 100, 100, 100, label_sz])
    y = tf.placeholder(tf.float32)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=model))
    optimiser = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
           loss = 0
           for batch_ind in range(nsamples // batch_sz):
               batch_x = data[batch_ind * batch_sz:(batch_ind + 1) * batch_sz]
               batch_y = labels[batch_ind * batch_sz:(batch_ind + 1) * batch_sz]
               _, c = sess.run([optimiser, cost], feed_dict={x:batch_x, y:batch_y})
               loss += c

           print(f'Epoch {epoch} completed. Epoch loss: {loss}')

        correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        test_accuracy = accuracy.eval({x:test_x, y:test_y})
        print(f'Accuracy: {test_accuracy}')



def make_data(nsample: int):
    ''' 
    Create some random data.
    Samples are three values [x1, x2, x3]
    The label of the sample is 1 if x3 is 'above' a plane defined by 
    x1 and x2, else 0
    Some noise is added to points near the plane (i.e. the class may be 'wrong')
    '''
    data = []
    labels = []
    for sn in range(nsample):
        x1 = random.randint(-100, 100)
        x2 = random.randint(-100, 100)
        x3a = 3 * x1 + 0.5 * x2 + 2 
        x3 = random.randint(int(x3a) - 100, int(x3a) + 100)
        noise = random.randint(-5, 5)

        # assign a class, one hot encoded
        # if 
        if x3 + noise > x3a:
            yl = [0, 1]
        else:
            yl = [1, 0]

        data.append([x1, x2, x3])
        labels.append(yl)

    data = np.array(data)
    labels = np.array(labels)
    return (data,labels)


data, labels = make_data(1000)
train_network(data, labels, 5, 100)
