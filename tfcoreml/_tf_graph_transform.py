

class Node(object):
  def __init__(self,
               type,
               inputs,
               outputs):

    self.type = type  # type: Text
    self.inputs = inputs  # type: List[Text]
    self.outputs = outputs  # type List[Text]
    self.parents = []  # type: List[Node]
    self.children = []  # type: List[Node]



class Graph(object):
  def __init__(self,
               nodes,
               root_node):

    self.nodes = nodes # type: List[Node]
    self.root = root_node # type: Node



def is_subgraph_match(tf_ops, # type: List[tf_ops]
                      root_id, # type: int
                      subgraph, # type: Graph
                      ):
  '''
  Find if the subgraph pattern exists in the TF graph starting from root_id op.
  :param tf_ops: list of TF ops
  :param root_id: subgraph is matched with the TF graph starting from tf_ops[root_id] 
  :param acyclic_graph: the subgraph corresponding to the pattern we want to match
  :return: bool: True if 'subgraph' found if tf_ops  
  '''

  subgraph_root = subgraph.root




