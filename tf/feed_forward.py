import numpy as np
import random
import tensorflow as tf

def simple(input_size, output_size):
    """
    Example of a simple fixed size feed forward neural network
    This has 2 hidden layers
    """
    # the layer sizes
    hidden_layer_1_size = 128
    hidden_layer_2_size = 128
    
    # create variables for each layers weights and biases
    layer1_weights = tf.Variable(tf.truncated_normal([input_size, hidden_layer_1_size]))
    layer1_biases = tf.Variable(tf.truncated_normal([hidden_layer_1_size]))
    layer2_weights = tf.Variable(tf.truncated_normal([hidden_layer_1_size, hidden_layer_2_size]))
    layer2_biases = tf.Variable(tf.truncated_normal([hidden_layer_2_size]))
    output_weights = tf.Variable(tf.truncated_normal([hidden_layer_2_size, output_size]))
    output_biases = tf.Variable(tf.truncated_normal([output_size]))
    
    # the placeholder for the input and output data
    x = tf.placeholder(tf.float32, shape=(None,input_size), name="X")
    y = tf.placeholder(tf.float32, shape=(None,output_size), name="Y")
    
    # construct the layers
    # the input is multiplied by layer1_weights, and then layer1_biases are added
    layer1 = tf.add(tf.matmul(x, layer1_weights), layer1_biases)
    # layer 1 non-linearity
    layer1 = tf.nn.relu(layer1)

    # layer 2 takes output of layer1
    layer2 = tf.add(tf.matmul(layer1, layer2_weights), layer2_biases)
    layer2 = tf.nn.relu(layer2)
                                
    # output layer takes output of layer2
    output = tf.add(tf.matmul(layer2, output_weights), output_biases)
    return x, y, output


def create_network(layer_sizes: list):
    """ 
    Construct a basic feed forward network from a list of layer sizes
    params:
        layer_sizes: The list of layer sizes to construct. This should include the input size (as the first value),
        and the output size (as the last value). E.g. [32, 512, 512, 10] would construct a network with an input size
        of 32, two hidden layers of size 512, and an output of size 10
    """
    layer_params = []
    data_size = layer_sizes[0]

    # create weights and bias parameters for all layers
    for cur, nxt in zip(layer_sizes[:-1],layer_sizes[1:]):
        wgts = tf.Variable(tf.random_normal([cur,nxt]))
        bias = tf.Variable(tf.random_normal([nxt]))
        layer_params.append({'weights':wgts,'bias':bias})

    # input data placeholder
    x = tf.placeholder(tf.float32, shape=(None,data_size))
    y = tf.placeholder(tf.float32)

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

    return x, y, cur_layer


def train_network(all_data, get_model, n_epochs, batch_sz):
    train_x, train_y, valid_x, valid_y = all_data

    sample_sz = len(train_x[0])
    label_sz = len(train_y[0])

    x, y, model = get_model()

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

        test_accuracy = accuracy.eval({x:valid_x, y:valid_y})
        print(f'Accuracy: {test_accuracy}')



def make_data(nsample: int):
    ''' 
    Create some random data.
    Samples are three values [x1, x2, x3]
    The label of the sample is 1 if x3 is 'above' a plane defined by 
    x1 and x2, else 0
    Some noise is added to points near the plane (i.e. the class may be 'wrong')
    '''
    # TODO - generate arrays directly using np.random
    data = []
    labels = []
    for sn in range(nsample):
        x1 = random.randint(-100, 100)
        x2 = random.randint(-100, 100)
        x3a = 3 * x1 + 0.5 * x2 + 2 
        x3 = random.randint(int(x3a) - 100, int(x3a) + 100)
        noise = random.randint(-5, 5)

        # assign a class, one hot encoded
        if x3 + noise > x3a:
            yl = [0, 1]
        else:
            yl = [1, 0]

        data.append([x1, x2, x3])
        labels.append(yl)

    data = np.array(data)
    labels = np.array(labels)
    return (data,labels)


if __name__ == "__main__":
    data, labels = make_data(1000)

    nsamples = len(data)
    nlabels = len(labels)
    if nsamples != nlabels:
        raise Exception("data and labels are differing sizes");w

    nsamples = int(nsamples * 0.8)
    train_x = data[:nsamples]
    train_y = labels[:nsamples]
    valid_x = data[nsamples+1:]
    valid_y = labels[nsamples+1:]
    all_data = (train_x, train_y, valid_x, valid_y)

    sample_sz = len(train_x[0])
    label_sz = len(train_y[0])

    use_simple = True
    if use_simple:
        create_network_func = lambda: simple(sample_sz, label_sz)
    else:
        create_network_func = lambda: create_network([sample_sz, 100, 100, 100, label_sz])

    train_network(all_data, create_network_func, 50, 100)
