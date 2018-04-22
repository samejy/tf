import numpy as np
import tensorflow as tf

def create_network(input_size, kernel_sizes: list, fc_layer_sizes: list):
    conv_layer_params = []
    imw, imh, imd = input_size
    
    last_conv_layer_size = [imw,imh,imd]

    for ksize in kernel_sizes:
        kw, kh, kd, n = ksize
        print(f'Creating conv layer weights/biases size {kw},{kh},{kd},{n}')
        # kernel size e.g. kw,kh = e.g. 5x5 kernel, kd = 1 channel?, n = number of kernels?
        wgts = tf.Variable(tf.random_normal([kw, kh, kd, n]))
        bias = tf.Variable(tf.random_normal([n])) # is this correct?
        conv_layer_params.append({'Weights':wgts, 'Bias':bias})
        # ?? 
        # maxpooling below will reduce by 2 
        last_conv_layer_size = [last_conv_layer_size[1]//2,last_conv_layer_size[2]//2,n] 

    inp_sz = last_conv_layer_size[0] * last_conv_layer_size[1] * last_conv_layer_size[2]
    fc_layer_params = []
    for fcsize in fc_layer_sizes:
        print(f'Creating fully connected layer input {inp_sz}, output {fcsize}')
        wgts = tf.Variable(tf.random_normal([inp_sz,fcsize]))
        bias = tf.Variable(tf.random_normal([fcsize]))
        fc_layer_params.append({'Weights':wgts, 'Bias':bias})
        inp_sz = fcsize
        
    # shape None is to allow any batch size?
    x = tf.placeholder(tf.float32, shape=(None,imw,imh,imd))

    # x = tf.reshape(x, ....) #???

    layer_input = x
    layers = []
    for cl_params in conv_layer_params:
        # TODO - how to do this without padding?
        l = tf.nn.conv2d(layer_input, cl_params['Weights'], [1,1,1,1], 'SAME') + cl_params['Bias']     
        l = tf.nn.relu(l)
        l = tf.nn.max_pool(l, [1,2,2,1], [1,2,2,1], 'SAME')
        layers.append(l)
        # becomes input for next layer
        layer_input = l

    layer_input = tf.reshape(layer_input, [-1,last_conv_layer_size[0] * last_conv_layer_size[1] * last_conv_layer_size[2]])

    n_fc_layers = len(fc_layer_params)
    for ind, fc_params in enumerate(fc_layer_params):
        l = tf.matmul(layer_input, fc_params['Weights']) + fc_params['Bias']
        if ind < n_fc_layers - 1:
            l = tf.nn.relu(l)
        layers.append(l)
        layer_input = l

    return x, layer_input
                           

# e.g. 7x7 kernel, 3 channels (?? to match input image?), 64 kernels
# e.g. 5x5 kernel, 64 channels, 128 kernels?
# 1024 nodes in fc layer, binary output

x, l = create_network((256,256,3), [(7,7,3,64), (5,5,64,128)], [1024,2]) 
        
                           
                           
    
