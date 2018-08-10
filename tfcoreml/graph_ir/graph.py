

class Graph(object):
    '''
    Class that defines the Graph Intermediate representation (IR). 
    Graph IR is used an intermediate format for converting the Tensforflow Graph to CoreML graph. 
    '''
    def __init__(self):
        self.nodes = [] # List[Node]
        self.inputs = [] # List[Tensor]
        self.tensor_map = {} # Dict[str, Tensor]. Maps tensor name to tensor object

    def make_graph_from_TF_ops(self, tf_ops): # [ops] -> Graph
        """
        Takes in the TF graph, represented as a list of ops and build the graph.
        
        :param tf_ops: List of TF operations
        """

        # add nodes to the graph
        for i, op in enumerate(tf_ops):
            node = Node(str(i) + '__' + op.name, op.type)
            for input_ in op.inputs:
                assert input_.name in self.tensor_map, ('source node not found for tensor called {}'.format(input_.name))
                tensor = self.tensor_map[input_.name]
                tensor.target_nodes.append(node)
                node.inputs.append(tensor)
            for output_ in op.outputs:
                tensor = Tensor(output_.name)
                tensor.source_node = node
                self.tensor_map[output_.name] = tensor
                node.outputs.append(tensor)
            self.nodes.append(node)

        # Lets add shapes to tensors
        for op in tf_ops:
            for output_ in op.outputs:
                shape = TensorShape()
                tf_shape = output_.get_shape()
                for s in tf_shape.as_list():
                    if s == None:
                        shape.shape.append(-1)
                    else:
                        shape.shape.append(s)
                shape.rank = len(tf_shape)
                shape.is_fully_defined = tf_shape.is_fully_defined()
                self.tensor_map[output_.name].shape = shape

class TensorShape(object):
    '''
    Class to represent the shape of a Tensor
    '''
    def __init__(self):
        self.is_fully_defined = False # Bool
        self.shape = [] # List[Int]
        self.rank = 0 # Int


class Tensor(object):
    '''
    Class to represent a Tensor 
    '''
    def __init__(self,
                 name = ''):
        self.name = name # str
        self.shape = None # TensorShape
        self.source_node = None # Node
        self.target_nodes = [] # List[Nodes]


class Node(object):
    '''
    Class that represents a node in the graph
    '''
    def __init__(self,
                 name = '',
                 type = ''):
        self.name = name #str
        self.type = type # str
        self.outputs = [] # List[Tensor]
        self.inputs = [] # List[Tensor]
        self.attributes = {} # Dict()



