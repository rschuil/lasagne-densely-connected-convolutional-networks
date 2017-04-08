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

    network = InputLayer(input_shape, input_var)

    network = Conv2DLayer(network,
        num_filters=num_filters_init,
        filter_size=(7,7),
        stride=(2,2),
        pad=(3,3),
        W=HeNormal(gain='relu'), b=None,
        flip_filters=False,
        nonlinearity=None)
    network = BatchNormLayer(network, beta=None, gamma=None)
    if dropout > 0:
      network = DropoutLayer(network, p=dropout)

    network = NonlinearityLayer(network, nonlinearity=rectify)
    network = MaxPool2DLayer(network,
      pool_size=(3,3),
      stride=(2,2),
      pad=(1,1))

    for i, num_layers in enumerate(stages):
        if i > 0:
            network = add_transition(network, math.floor(network.output_shape[1] / 2), dropout)
        network = build_block(network, num_layers, growth_rate, dropout)

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
      
    layer = ConcatLayer([incoming, layer], axis=1)

    return layer

def add_transition(incoming, num_filters, dropout):
    layer = ScaleLayer(incoming)
    layer = BiasLayer(layer)
    layer = NonlinearityLayer(layer, nonlinearity=rectify)
    layer = Conv2DLayer(layer,
        num_filters=num_filters,
        filter_size=(1,1),
        stride=(1,1),
        W=HeNormal(gain='relu'), b=None,
        flip_filters=False,
        nonlinearity=None)
    if dropout > 0:
        layer = DropoutLayer(layer, p=dropout)
    layer = Pool2DLayer(layer,
        pool_size=(2,2),
        stride=(2,2),
        mode='average_exc_pad')
    layer = BatchNormLayer(layer, beta=None, gamma=None)
    return layer
