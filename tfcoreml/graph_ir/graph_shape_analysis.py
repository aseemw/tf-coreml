import copy
import collections

def add_op(gc, node, forward=True):
    # https://www.tensorflow.org/api_docs/python/tf/add
    # 2 inputs, 1 output
    if forward:
        if node.outputs[0].shape.rank == node.inputs[0].shape.rank:
            node.outputs[0].shape.shape = copy.copy(node.inputs[0].shape.shape)
        elif node.outputs[0].shape.rank == node.inputs[1].shape.rank:
            node.outputs[0].shape.shape = copy.copy(node.inputs[1].shape.shape)
        else:
            raise ValueError('Add: output rank must match one of the inputs rank')
    else:
        for i in range(2):
            in_r = node.inputs[i].shape.rank
            node.inputs[i].shape.shape[:] = node.outputs[0].shape.shape[-in_r:]

def matmul_op(gc, node, forward=True):
    # https://www.tensorflow.org/api_docs/python/tf/matmul
    # 2 inputs, 1 output
    assert node.inputs[0].shape.rank == node.inputs[1].shape.rank
    assert node.inputs[0].shape.rank == node.outputs[0].shape.rank
    if forward:
        node.outputs[0].shape.shape = copy.copy(node.inputs[0].shape.shape)
    else:
        node.inputs[0].shape.shape = copy.copy(node.outputs[0].shape.shape)
        node.inputs[1].shape.shape = copy.copy(node.outputs[0].shape.shape)


OPS_TYPES_REGISTRY = {
    'Add': add_op,
    'MatMul': matmul_op,
}


def voting_algo(gc, voting_dict):

    '''
    Based on the type of node, vot for what label the dimensions should get.
    Voting can be strong/hard or weak/soft. 
    
    :param gc: 
    :param voting_dict: 
    :return: 
    '''

    def _get_shape(gc, edge_name):
        if edge_name in gc.edge_equivalence_map:
            edge = gc.edge_equivalence_map[edge_name]
        else:
            edge = edge_name
        if edge in gc.shape_compressed_graph.tensor_map:
            return gc.shape_compressed_graph.tensor_map[edge].shape.shape
        else:
            return None

    def _get_voting_dict_bins(shape):
        x, y = [int(j) for j in shape.split('_')]
        return x, y


    # here we will iterate over the raw graph nodes
    for node in gc.raw_graph.nodes:
        if node.type in ['MaxPool', 'Conv2D']:
            s = _get_shape(gc, node.outputs[0].name)
            if s:
                for i, l in enumerate(['B', 'H', 'W', 'C']):
                    x, y = _get_voting_dict_bins(s[i])
                    voting_dict[x][y][0].append(l)
        if node.type in ['MatMul']:
            s = _get_shape(gc, node.outputs[0].name)
            if s:
                x, y = _get_voting_dict_bins(s[0])
                voting_dict[x][y][1].append('B')
                if len(s) == 2:
                    x, y = _get_voting_dict_bins(s[1])
                    voting_dict[x][y][1].append('C')


class Shape_analysis(object):
    def __init__(self, graph_collections):
        self.gc = graph_collections

    def _get_shape_function(self, node_type):
        if node_type in OPS_TYPES_REGISTRY:
            return OPS_TYPES_REGISTRY[node_type]
        else:
            raise ValueError("shape function not found for node of type {}".format(node_type))

    def make_shape_dict(self, print_info=False):
        shape_dict = {}
        partially_unresolved = 0
        full_unresolved = 0
        for k,v in self.gc.shape_compressed_graph.tensor_map.items():
            shape_dict[k] = v.shape.shape
            n_minus_ones = v.shape.shape.count(-1)
            if n_minus_ones == len(v.shape.shape):
                full_unresolved += 1
            elif n_minus_ones > 0:
                partially_unresolved += 1
        if print_info:
            n = len(shape_dict)
            print("Out of {} tensors, fully unresolved: {}, partially unresolved = {}".format(n, full_unresolved, partially_unresolved))
        return shape_dict


    def run_shape_analysis(self, run_voting_algorithm=True):
        n = 0
        graph = self.gc.shape_compressed_graph
        voting_dict = collections.OrderedDict()
        # populate the shapes of 'placeholders' i.e. inputs
        for node in graph.nodes:
            if node.type == 'Placeholder':
                voting_dict[n] = collections.OrderedDict()
                out = node.outputs[0]
                for r in range(out.shape.rank):
                    out.shape.shape[r] = str(n)+'_'+str(r)
                    voting_dict[n][r] = [[],[]] # list of two lists: strong voting, weak voting
                n += 1
                break

        # run shape analysis forward, backward, until no change in shapes
        break_loop = False
        while not break_loop:
            shape_dict_1 = self.make_shape_dict()
            #forward pass
            for node in graph.nodes:
                if len(node.inputs) > 0 and len(node.outputs) > 0:
                    foo = self._get_shape_function(node.type)
                    foo(self.gc, node, forward=True)
            #backward pass
            for node in reversed(graph.nodes):
                if len(node.inputs) > 0 and len(node.outputs) > 0:
                    foo = self._get_shape_function(node.type)
                    foo(self.gc, node, forward=False)
            shape_dict_2 = self.make_shape_dict()
            if shape_dict_1 == shape_dict_2:
                break_loop = True

        # print info about how many tensors were left out
        shape_dict = self.make_shape_dict(print_info=True)

        if run_voting_algorithm:

            # do the voting step
            voting_algo(self.gc, voting_dict)

            # complete voting dictionary
            for kv,v in voting_dict.items():
                for kz, z in v.items():
                    hard_votes = z[0] # List[str]
                    soft_votes = z[1] # List[str]
                    if len(hard_votes) > 0:
                        assert len(set(hard_votes)) == 1, "inconsistent voting"
                        voting_dict[kv][kz][0] = [hard_votes[0]]
                    else:
                        assert len(soft_votes) > 0, "no hard votes and no soft votes"
                        # find mode from the soft voting
                        voting_dict[kv][kz][0] = max(set(soft_votes), key=soft_votes.count)


            # populate tensor labeled shapes
            # first for the compressed graph
            for tname, t in self.gc.shape_compressed_graph.tensor_map.items():
                for s in t.shape.shape:
                    x, y = [int(j) for j in s.split('_')]
                    t.shape.labeled_shape.append(voting_dict[x][y][0][0])
            # now for the raw graph
            for tname, t in self.gc.raw_graph.tensor_map.items():
                if tname in self.gc.edge_equivalence_map:
                    edge = self.gc.edge_equivalence_map[tname]
                else:
                    edge = tname
                if edge in self.gc.shape_compressed_graph.tensor_map:
                    ls = self.gc.shape_compressed_graph.tensor_map[edge].shape.labeled_shape
                    t.shape.labeled_shape = copy.copy(ls)




