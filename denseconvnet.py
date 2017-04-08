import math
from lasagne.layers import InputLayer, Conv2DLayer, NonlinearityLayer, Pool2DLayer, MaxPool2DLayer, ScaleLayer, BiasLayer, ConcatLayer, DropoutLayer, GlobalPoolLayer, DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.init import HeNormal
try:
    from lasagne.layers.dnn import BatchNormDNNLayer as BatchNormLayer
except ImportError:
    from lasagne.layers import BatchNormLayer

'''
Borrows heavily from "Densely Connected Convolutional Networks (DenseNets)" (https://github.com/liuzhuang13/DenseNet)
and "Densely Connected Convolutional Network (DenseNet) in Lasagne" (https://github.com/Lasagne/Recipes/tree/master/papers/densenet).
'''

def build_densenet(
        input_var,
        input_shape=(None, 3, 224, 224),
        num_filters_init=64,
        growth_rate=32,
        dropout=0.2,
        num_classes=1000,
        stages=[6,12,24,16]):

    if input_shape[2] % (2 ** len(stages)) != 0:
      raise ValueError("input_shape[2] must be a multiple of {}.".format(2 ** len(stages)))

    if input_shape[3] % (2 ** len(stages)) != 0:
      raise ValueError("input_shape[3] must be a multiple of {}.".format(2 ** len(stages)))      

    # Input should be (BATCH_SIZE, NUM_CHANNELS, WIDTH, HEIGHT)
    # NUM_CHANNELS is usually 3 (R,G,B) and for the ImageNet example the width and height are 224
    network = InputLayer(input_shape, input_var)

    # Apply 2D convolutions with a 7x7 filter (pad by 3 on each side)
    # Because of the 2x2 stride the shape of the last two dimensions will be half the size of the input (112x112)
    network = Conv2DLayer(network,
        num_filters=num_filters_init,
        filter_size=(7,7),
        stride=(2,2),
        pad=(3,3),
        W=HeNormal(gain='relu'), b=None,
        flip_filters=False,
        nonlinearity=None)
    
    # Batch normalize
    network = BatchNormLayer(network, beta=None, gamma=None)
    
    # If dropout is enabled, apply after every convolutional and dense layer
    if dropout > 0:
      network = DropoutLayer(network, p=dropout)

    # Apply ReLU
    network = NonlinearityLayer(network, nonlinearity=rectify)
    
    # Keep the maximum value of a 3x3 pool with a 2x2 stride
    # This operation again divides the size of the last two dimensions by two (56x56)
    network = MaxPool2DLayer(network,
      pool_size=(3,3),
      stride=(2,2),
      pad=(1,1))

    # Add dense blocks
    for i, num_layers in enumerate(stages):
        # Except for the first block, we add a transition layer before the dense block that halves the number of filters, width and height
        if i > 0:
            network = add_transition(network, math.floor(network.output_shape[1] / 2), dropout)
        network = build_block(network, num_layers, growth_rate, dropout)

    # Apply global pooling and add a fully connected layer with softmax function
    network = ScaleLayer(network)
    network = BiasLayer(network)
    network = NonlinearityLayer(network, nonlinearity=rectify)
    network = GlobalPoolLayer(network)
    network = DenseLayer(network,
        num_units=num_classes,
        W=HeNormal(gain=1),
        nonlinearity=softmax)

    return network

def build_block(incoming, num_layers, num_channels, dropout):
    layer = incoming
    for _ in range(num_layers):
        layer = add_layer(layer, num_channels, dropout)
    return layer

def add_layer(incoming, num_channels, dropout):
    layer = ScaleLayer(incoming)
    layer = BiasLayer(layer)
    
    # Bottleneck layer to reduce number of input channels to 4 times the number of output channels
    layer = NonlinearityLayer(layer, nonlinearity=rectify)
    layer = Conv2DLayer(layer,
        num_filters=4*num_channels,
        filter_size=(1,1),
        stride=(1,1),
        W=HeNormal(gain='relu'), b=None,
        flip_filters=False,
        nonlinearity=None)
    layer = BatchNormLayer(layer, beta=None, gamma=None)
    if dropout > 0:
        layer = DropoutLayer(layer, p=dropout)
    
    # Convolutional layer (using padding to keep same dimensions)
    layer = NonlinearityLayer(layer, nonlinearity=rectify)
    layer = Conv2DLayer(layer,
        num_filters=num_channels,
        filter_size=(3,3),
        stride=(1,1),
        W=HeNormal(gain='relu'), b=None,
        pad='same',
        flip_filters=False,
        nonlinearity=None)        
    layer = BatchNormLayer(layer, beta=None, gamma=None)
    if dropout > 0:
      layer = DropoutLayer(layer, p=dropout)

    # Concatenate the input filters with the new filters
    layer = ConcatLayer([incoming, layer], axis=1)

    return layer

def add_transition(incoming, num_filters, dropout):
    layer = ScaleLayer(incoming)
    layer = BiasLayer(layer)
    layer = NonlinearityLayer(layer, nonlinearity=rectify)
    # Reduce the number of filters
    layer = Conv2DLayer(layer,
        num_filters=num_filters,
        filter_size=(1,1),
        stride=(1,1),
        W=HeNormal(gain='relu'), b=None,
        flip_filters=False,
        nonlinearity=None)
    if dropout > 0:
        layer = DropoutLayer(layer, p=dropout)
    # Pooling layer to reduce the last two dimensions by half
    layer = Pool2DLayer(layer,
        pool_size=(2,2),
        stride=(2,2),
        mode='average_exc_pad')
    layer = BatchNormLayer(layer, beta=None, gamma=None)
    return layer
