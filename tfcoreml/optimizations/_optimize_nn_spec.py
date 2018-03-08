from . import _optimize

def _optimize_fold_load_constants(nn_layers):
  """
  Fold load constants that interact through 'add', 'multiply', 'activation',
  'slice', 'reduce' or 'unary' layers.
  In other words, evaluate any sub-graph that involves only 'load_constant',
  'multiply', 'add', 'activation', 'slice', 'reduce'
  or 'unary' layer types and replace it with a single load constant layer.
  """

  _optimize._fold_constants(nn_layers)

def _optimize_conv_mul_add(nn_layers):
  """
  Detect Multiply or add layers after convolution and recast as Batchnorm layer
  so that it can be fused in the framework.
  """
  _optimize._fuse_conv_mul_add(nn_layers)


def _optimize_spatial_reduce_operation(nn_layers):
  """
  Find a reduce layer with mode = 'average'/'max' and axis = 'HW'
  and replace it with global average/max pooling layer.
  """

  _optimize._spatial_reduce_as_global_pool(nn_layers)


def _optimize_disconnected_components(spec, nn_spec):
  """
  Removes from the CoreML NN graph all the layers that are not connected
  to the output nodes.
  """

  _optimize._remove_disconnected_components(spec, nn_spec)

def _optimize_pad_conv(nn_layers):
  """
  Fuses pad-conv layers when the pad layer is doing 'constant' padding with zeros
  """
  _optimize._fuse_pad_conv(nn_layers)


def optimize_nn_spec(spec):
  """
  Call a specific set of network optimizations
  """

  if spec.WhichOneof('Type') == 'neuralNetwork':
    nn_spec = spec.neuralNetwork
  elif spec.WhichOneof('Type') == 'neuralNetworkClassifier':
    nn_spec = spec.neuralNetworkClassifier
  elif spec.WhichOneof('Type') == 'neuralNetworkRegressor':
    nn_spec = spec.neuralNetworkRegressor
  else:
    raise ValueError('Specification must contain a neural network')

  _optimize_fold_load_constants(nn_spec.layers)
  _optimize_spatial_reduce_operation(nn_spec.layers)
  _optimize_pad_conv(nn_spec.layers)
  _optimize_conv_mul_add(nn_spec.layers)
  _optimize_disconnected_components(spec, nn_spec)


