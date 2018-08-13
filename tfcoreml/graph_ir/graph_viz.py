import os

DEBUG = False


def plot_graph(graph,
               graph_img_path='graph.png'):
    """
    Plot graph using pydot, the library the Keras uses as well for visualization. 
    
    It works in two steps:
    1. Add nodes to pydot
    2. connect nodes added in pydot
    
    :param graph: 
    :return: writes down a png/pdf file using dot 
    """

    try:
        # pydot-ng is a fork of pydot that is better maintained.
        import pydot_ng as pydot
    except:
        # pydotplus is an improved version of pydot
        try:
            import pydotplus as pydot
        except:
            # Fall back on pydot if necessary.
            try:
                import pydot
            except:
                return None

    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')



    if DEBUG:
        print('-' * 200)
        for node in graph.nodes:
            print('Node type: {}, name: {},  input name: {}, output names = {}'.format(node.type,node.name,
                                                               str([in_.name for in_ in node.inputs]),
                                                               str([out_.name for out_ in node.outputs])))
        for t_name, t in graph.tensor_map.iteritems():
            print('tensor name: {}, source = {}, target = {}'.format(t_name,
                                                                 t.source_node,
                                                                 str([tn for tn in t.target_nodes])))

    # Traverse graph and add nodes to pydot
    for node in graph.nodes:
        inputlabels = ', '.join(
            [str(tuple(input_.shape.shape)) for input_ in node.inputs])
        outputlabels = ', '.join(
            [str(tuple(output_.shape.shape)) for output_ in node.outputs])
        input_names = ', '.join([input_.name for input_ in node.inputs])
        output_names = ', '.join([output_.name for output_ in node.outputs])
        label = label = '%s\n|{|%s}|{{%s}|{%s}}' % (node.name.split('___')[0] + '_' + node.type,
                                                                        output_names,
                                                                        inputlabels,
                                                                        outputlabels)
        pydot_node = pydot.Node(node.name, label=label)
        dot.add_node(pydot_node)

    # add edges
    for node in graph.nodes:
        for output_ in node.outputs:
            for target_ in output_.target_nodes:
                # add edge in pydot
                if DEBUG:
                    print('adding edge between node {} and node {}'.format(node.name, target_))
                dot.add_edge(pydot.Edge(node.name, target_))

    # write out the image file
    _, extension = os.path.splitext(graph_img_path)
    if not extension:
        extension = 'pdf'
    else:
        extension = extension[1:]
    dot.write(graph_img_path, format=extension)



