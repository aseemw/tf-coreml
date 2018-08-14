RANK_PRESERVING_OPS = ['Identity', 'Relu', 'MaxPool', 'Conv2D']

DEBUG = False

import copy

class TensorShape(object):
    '''
    Class to represent the shape of a Tensor
    '''

    def __init__(self):
        self.is_fully_defined = False  # Bool
        self.shape = []  # List[Int]
        self.rank = 0  # Int
        self.labeled_shape = [] # List[str]

class Tensor(object):
    '''
    Class to represent a Tensor 
    '''

    def __init__(self,
                 name='',
                 shape = None):
        self.name = name  # str
        self.shape = copy.deepcopy(shape)  # TensorShape
        self.source_node = ''  # str
        self.target_nodes = []  # List[str]

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

class Graph(object):
    '''
    Class that defines the Graph Intermediate representation (IR). 
    Graph IR is used an intermediate format for converting the Tensforflow Graph to CoreML graph. 
    '''
    def __init__(self):
        self.nodes = [] # List[Node]

        self.inputs = []  # List[Tensor], as of now unused
        self.tensor_map = {} # Dict[str, Tensor]. Maps tensor name to tensor object, need it to build the graph, useful to have.
        self.node_map = {} # Dict[str, Node]. Maps node name to Node object, useful to have.


    def max_rank(self):
        r = 0
        for _,t in self.tensor_map.items():
            r = max(r, t.shape.rank)
        return r

    def update_tensor_source_target_info(self):

        if DEBUG:
            print('-' * 200)
            for node in self.nodes:
                print('Node type: {}, name: {},  input name: {}, output names = {}'.format(node.type, node.name,
                                                                                           str([in_.name for in_ in
                                                                                                node.inputs]),
                                                                                           str([out_.name for out_ in
                                                                                                node.outputs])))
        for tensor_name, tensor in self.tensor_map.items():
            tensor.source_node = None
            tensor.target_nodes = []
        for node in self.nodes:
            self.node_map[node.name] = node
        for node in self.nodes:
            for input_ in node.inputs:
                input_.target_nodes.append(node.name)
            for output_ in node.outputs:
                output_.source_node = node.name


    def make_graph_from_TF_ops(self, tf_ops): # [ops] -> Graph
        """
        Takes in the TF graph, represented as a list of ops and build the graph.
        :param tf_ops: List of TF operations
        """

        # add nodes to the graph
        for i, op in enumerate(tf_ops):
            node = Node(str(i) + '___' + op.name, op.type)
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



class GraphCollections(object):
    '''
    Hold several variants of the same graph (one raw and other optimized versions)
    '''
    def __init__(self):
        self.raw_graph = Graph() # Graph
        self.tf_ops = [] # List[TF operations]
        self.shape_compressed_graph = Graph() # Graph

        '''
        The edges/tensors in the raw_graph can be divided into 3 classes:
        1. the ones that are also present in the shape_compressed_graph
        2. the ones that are not present in the shape_compressed_graph, but can be mapped to
            one of the tensors that is present. The edge_equivalence_map stores this information
        3. the ones that are not present and not mapped to any tensor in shape_compressed_graph, 
           these are the ones that may correspond to weight/other params in an op and their shape 
           might not be important. 
        '''
        self.edge_equivalence_map = dict() # Dict[str, str]


    def build_compressed_graph(self):
        self.shape_compressed_graph = Graph() # clear the graph, we are going to build it fresh
        raw_graph = copy.deepcopy(self.raw_graph) # copy just in case we don't want to mess with original graph

        # iterate over the nodes and add them to the new graph
        for node in raw_graph.nodes:
            if node.type in RANK_PRESERVING_OPS:
                self.edge_equivalence_map[node.outputs[0].name] = self.edge_equivalence_map.get(node.inputs[0].name, node.inputs[0].name)
            else:
                new_node = Node(node.name, node.type)
                for out_edge in node.outputs:
                    tensor = Tensor(out_edge.name, out_edge.shape)
                    tensor.shape.shape = [-1 for i in range(out_edge.shape.rank)]
                    tensor.shape.rank = out_edge.shape.rank
                    self.shape_compressed_graph.tensor_map[tensor.name] = tensor
                    new_node.outputs.append(tensor)
                for inbound_edge in node.inputs:
                    if inbound_edge.name in self.edge_equivalence_map:
                        name = self.edge_equivalence_map[inbound_edge.name]
                    else:
                        name = inbound_edge.name
                    assert name in self.shape_compressed_graph.tensor_map, ('source node not found for tensor called {}'.format(name))
                    new_node.inputs.append(self.shape_compressed_graph.tensor_map[name])
                self.shape_compressed_graph.nodes.append(new_node)

        # remove singleton disconnected nodes
        inbound_edges_in_use = {}
        outbound_edges_in_use = {}
        for node in self.shape_compressed_graph.nodes:
            for in_ in node.inputs:
                inbound_edges_in_use[in_.name] = 1
            for out_ in node.outputs:
                outbound_edges_in_use[out_.name] = 1
        node_ids_marked_for_removal = []
        for i, node in enumerate(self.shape_compressed_graph.nodes):
            to_be_removed = True
            for in_ in node.inputs:
                if in_.name in outbound_edges_in_use:
                    to_be_removed = False
                    break
            for out_ in node.outputs:
                if out_.name in inbound_edges_in_use:
                    to_be_removed = False
                    break
            if to_be_removed: node_ids_marked_for_removal.append(i)
        for index in sorted(node_ids_marked_for_removal, reverse=True):
            node = self.shape_compressed_graph.nodes[index]
            for in_ in node.inputs:
                if in_.name in self.shape_compressed_graph.tensor_map:
                    del self.shape_compressed_graph.tensor_map[in_.name]
            for out_ in node.outputs:
                if out_.name in self.shape_compressed_graph.tensor_map:
                    del self.shape_compressed_graph.tensor_map[out_.name]
            del self.shape_compressed_graph.nodes[index]

        self.shape_compressed_graph.update_tensor_source_target_info()

