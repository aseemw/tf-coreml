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
        out_r = node.outputs[0].shape.rank
        for i in range(2):
            in_r = node.inputs[i].shape.rank
            if out_r == in_r:
                node.inputs[i].shape.shape = copy.copy(node.outputs[0].shape.shape)
            else:
                assert in_r < out_r
                node.inputs[i].shape.shape[:] = node.outputs[0].shape.shape[-in_r:]


OPS_TYPES_REGISTRY = {
    'Add': add_op,
}



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


    def run_shape_analysis(self):
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

        # do the voting step
        # here we will iterate over the raw graph nodes
        for node in self.gc.raw_graph.nodes:
            if node.type in ['MaxPool', 'Conv2D']:
                out_ = node.outputs[0].name
                if out_ in self.gc.edge_equivalence_map:
                    edge = self.gc.edge_equivalence_map[out_]
                else:
                    edge = out_
                if edge in self.gc.shape_compressed_graph.tensor_map:
                    s = self.gc.shape_compressed_graph.tensor_map[edge].shape.shape
                    for i,l in enumerate(['B','H','W','C']):
                        x, y = [int(j) for j in s[i].split('_')]
                        voting_dict[x][y][0].append(l)

        # complete voting dictionary
        for kv,v in voting_dict.items():
            for kz, z in v.items():
                svotes = z[0]
                assert len(svotes) > 0, "no votes"
                assert len(set(svotes)) == 1, "inconsistent voting"
                voting_dict[kv][kz][0] = [z[0][0]]

        # populate tensor labeled shapes

        # first for the compressed graph
        for tname, t in self.gc.shape_compressed_graph.tensor_map.items():
            for s in t.shape.shape:
                x, y = [int(j) for j in s.split('_')]
                t.shape.labeled_shape.append(voting_dict[x][y][0][0])
        for tname, t in self.gc.raw_graph.tensor_map.items():
            if tname in self.gc.edge_equivalence_map:
                edge = self.gc.edge_equivalence_map[tname]
            else:
                edge = tname
            if edge in self.gc.shape_compressed_graph.tensor_map:
                ls = self.gc.shape_compressed_graph.tensor_map[edge].shape.labeled_shape
                t.shape.labeled_shape = copy.copy(ls)




