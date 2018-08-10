import os


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

    # Traverse graph and add nodes to pydot
    for node in graph.nodes:
        inputlabels = ', '.join(
            [str(tuple(input_.shape.shape)) for input_ in node.inputs])
        outputlabels = ', '.join(
            [str(tuple(output_.shape.shape)) for output_ in node.outputs])
        label = label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (node.type,
                                                           inputlabels,
                                                           outputlabels)
        pydot_node = pydot.Node(node.name, label=label)
        dot.add_node(pydot_node)

    # add edges
    for node in graph.nodes:
        for output_ in node.outputs:
            for target_ in output_.target_nodes:
                # add edge in pydot
                dot.add_edge(pydot.Edge(node.name, target_.name))

    # write out the image file
    _, extension = os.path.splitext(graph_img_path)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(graph_img_path, format=extension)



