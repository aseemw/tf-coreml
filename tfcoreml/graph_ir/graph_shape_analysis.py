import copy

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

    def make_shape_dict(self):
        shape_dict = {}
        for k,v in self.gc.shape_compressed_graph.tensor_map.iteritems():
            shape_dict[k] = v.shape.shape
        return shape_dict


    def run_shape_analysis(self):
        n = 0
        graph = self.gc.shape_compressed_graph
        # populate the shapes of 'placeholders' i.e. inputs
        for node in graph.nodes:
            if node.type == 'Placeholder':
                out = node.outputs[0]
                for r in range(out.shape.rank):
                    out.shape.shape[r] = str(n)+'_'+str(r)
            n+=1

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

