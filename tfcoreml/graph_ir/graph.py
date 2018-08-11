OP_SKIP_LIST = ['Const', 'Identity']
import copy

class TensorShape(object):
    '''
    Class to represent the shape of a Tensor
    '''

    def __init__(self):
        self.is_fully_defined = False  # Bool
        self.shape = []  # List[Int]
        self.rank = 0  # Int

class Tensor(object):
    '''
    Class to represent a Tensor 
    '''

    def __init__(self,
                 name=''):
        self.name = name  # str
        self.shape = None  # TensorShape
        self.source_node = None  # Node
        self.target_nodes = []  # List[Nodes]

class Node(object):
    '''
    Class that represents a node in the graph
    '''

    def __init__(self,
                 name='',
                 type=''):
        self.name = name  # str
        self.type = type  # str
        self.outputs = []  # List[Tensor]
        self.inputs = []  # List[Tensor]
        self.attributes = {}  # Dict()



class GraphCollections(object):
    '''
    Hold several variants of the same graph (one raw and other optimized versions)
    '''
    def __init__(self):
        self.graph_main = Graph() # Graph
        self.graph_compressed = Graph() # Graph

        # Keys are the tensor names that exist in the main graph, but not in the compressed garph.
        # Their corresponding tensors in the compressed graph are the values.
        self.edge_equivalence_map = dict() # Dict[str, str].


    def build_compressed_graph(self):
        self.graph_compressed = Graph() # clear the graph
        # iterate over the the main graph and build edge (tensor) equivalence map
        for node in self.graph_main.nodes:
            n_inputs = len(node.inputs)
            if node.type in OP_SKIP_LIST and (n_inputs == 0 or n_inputs == 1):
                for out_edge in node.outputs:
                    self.edge_equivalence_map[out_edge.name] = None if n_inputs == 0 \
                                                               else self.edge_equivalence_map.get(node.inputs[0].name, node.inputs[0].name)
            else:
                new_node = copy.deepcopy(node)
                new_node.inputs = []
                for out_edge in node.outputs:
                    self.graph_compressed.tensor_map[out_edge.name] = out_edge
                for inbound_edge in node.inputs:
                    if inbound_edge.name in self.edge_equivalence_map:
                        edge_name = self.edge_equivalence_map[inbound_edge.name]
                        if edge_name is not None:
                            edge = self.graph_main.tensor_map[edge_name]
                            new_node.inputs.append(edge)
                    else:
                        new_node.inputs.append(inbound_edge)
                self.graph_compressed.nodes.append(new_node)

        self.graph_compressed.update_tensor_source_target_info()


class Graph(object):
    '''
    Class that defines the Graph Intermediate representation (IR). 
    Graph IR is used an intermediate format for converting the Tensforflow Graph to CoreML graph. 
    '''
    def __init__(self):
        self.nodes = [] # List[Node]

        self.inputs = []  # List[Tensor], as of now unused
        self.tensor_map = {} # Dict[str, Tensor]. Maps tensor name to tensor object, need it to build the graph, useful to have.


    def update_tensor_source_target_info(self):
        for tensor_name, tensor in self.tensor_map.iteritems():
            tensor.source_node = None
            tensor.target_nodes = []
        for node in self.nodes:
            for input_ in node.inputs:
                input_.target_nodes.append(node)
            for output_ in node.outputs:
                output_.source_node = node


    def make_graph_from_TF_ops(self, tf_ops): # [ops] -> Graph
        """
        Takes in the TF graph, represented as a list of ops and build the graph.
        :param tf_ops: List of TF operations
        """

        # add nodes to the graph
        for i, op in enumerate(tf_ops):
            node = Node(str(i) + '__' + op.name, op.type)
            for input_ in op.inputs:
                # since its an input, the tensor should already been have added
                assert input_.name in self.tensor_map, ('source node not found for tensor called {}'.format(input_.name))
                node.inputs.append(self.tensor_map[input_.name])
            for output_ in op.outputs:
                # create a new tensor
                tensor = Tensor(output_.name)
                # add the tensor to the list of tensors/edges that we want to maintain
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


        # update source/target info in tensors
        self.update_tensor_source_target_info()



