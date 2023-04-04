"""
Module has functions and classes to parse through NextStrain Auspice Tree data
see the schema : https://github.com/nextstrain/augur/blob/master/augur/data/schema-export-v2.json
"""
import pandas as pd
from datetime import datetime, timedelta
from MutationHelpers import mutate_sequence, sort_mut_list, aln_parent_child_node, pullout_pos_mut_df, \
    node_muts_from_ref
import json
from Bio import SeqIO
import pickle
from tqdm import tqdm


def has_children(node):
    """
    Helper function to determine if a given node has children. Children are contained in list with key = children
    :param node: node data from auspice tree
    :type node: dict
    :return: True if node has children, false otherwise
    :rtype: bool
    """
    try:
        children_list = node['children']
    except KeyError:
        return False
    return True


def node_connections(node):
    """
    Function to get the connection child node names from parent node
    :param node: node data from auspice tree
    :type node: dict
    :return: list of [parent, child]
    :rtype: list
    """
    node_edges = []
    parent_name = node['name']
    for child in node['children']:
        child_name = child['name']
        node_edges.append((parent_name, child_name))
    return node_edges


def tree_edge_list(node, edge_list=[]):
    """
    Recursive function to get edge list from auspice tree
    finds the node connections of parent node then recurses down to children nodes. Can call from root
    :param node: node data from auspice tree
    :type node: dict
    :param edge_list: list of edges
    :type edge_list: list
    :return: full edge list from tree
    :rtype: list
    """
    if has_children(node):
        node_edges = node_connections(node)
        edge_list.extend(node_edges)
        for child in node['children']:
            tree_edge_list(child, edge_list=edge_list)
    return edge_list


def paths_dict(node, ancestors=[], allPaths={}):
    """
    Recursive function to get paths from root to all leaf nodes in a dictionary of key = leaf, value = path
    start at root to get all paths of tree
    :param node: node data from auspice tree
    :type node: dict
    :param ancestors: current list of ancestors of node
    :type ancestors: list
    :param allPaths: current dictionary of key=leaf, value=path
    :type allPaths: dict
    :return: full dictionary of key=leaf, value=path
    :rtype: dict
    """
    ancestors.append(node['name'])
    if not has_children(node):
        allPaths[node['name']] = ancestors
    else:
        for child in node['children']:
            paths_dict(child, ancestors[:], allPaths)
    return allPaths


def ancestors_from_edgelist(nodename, edge_df, root_node='NODE_0000001'):
    """
    function to get all ancestors of a given node from a dataframe of (parent, child) relationships from tree
    :param nodename: node of interest
    :type nodename: str
    :param edge_df: edge list in dataframe
    :type edge_df: pandas.DataFrame
    :param root_node: root node name
    :type root_node: str
    :return: list of ancestors from given node to root
    :rtype: list
    """
    ancestors = []
    if nodename == root_node:
        return ancestors
    ancestor = edge_df[(edge_df['Child'] == nodename)]['Parent'].values[0]
    while ancestor != root_node:
        ancestors.append(ancestor)
        ancestor = edge_df[(edge_df['Child']==ancestor)]['Parent'].values[0]
    ancestors.append(root_node)
    ancestors.reverse()
    return ancestors


def tree_node_data(node, parent=None, node_data_dict={}):
    """
    Recursive Function to get a dictionary of key=node name, value = Node class
    creates node class from current node and recurses down to children nodes
    :param node: node data from auspice tree
    :type node: dict
    :param parent: name of parent node
    :type parent: str
    :param node_data_dict: current dictionary holding tree node data
    :type node_data_dict: dict
    :return: full tree node data dictionary, key=node name, value=Node()
    :rtype: dict
    """
    node_class = Node()
    node_class.initialize(node_dict=node,  parent=parent)
    node_data_dict[node_class.name] = node_class
    if has_children(node):
        parent_name = node['name']
        for child in node['children']:
            tree_node_data(node=child, parent=parent_name, node_data_dict=node_data_dict)
    return node_data_dict


def get_date_from_num(start):
    """
    Function converts a number date to python datetime datetime
    example: 2021.9328767123288 = 12/7/2021
    :param start: number date
    :type start: float
    :return: datetime date
    :rtype: datetime.datetime
    """
    year = int(start)
    rem = start - year
    base = datetime(year, 1, 1)
    result = base + timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
    return result


class Node(object):
    """
    class to hold Node Data

    Attributes
    ----------
    name : str
        name of node
    parent : str
        name of parent node
    root : bool
        true if root, false otherwise
    children : list
        list of children nodes
    branch_attrs : dict
        dictionary of branch attributes
    node_attrs : dict
        dictionary of node attributes
    leaf : bool
        true if leaf, false otherwise
    path : list
        list of nodes starting from parent [0], to root [-1]. if node is root, empty list
    spike_mutations : list
        list of spike amino acid mutations inherited from root
    spike_seq : str
        mutated spike sequence from list of mutations
    node_spike_mutations : list
        list of mutations between parent and child node

    """
    def __init__(self):
        self.name = None
        self.parent = None
        self.root = False
        self.children = []
        self.branch_attrs = None
        self.node_attrs = None
        self.date = None
        self.leaf = False
        self.path = []
        # list to hold all spike mutations from reference to node
        self.spike_mutations = []
        self.spike_seq = None
        # list to hold mutations from parent-child
        self.node_spike_mutations = []
        # list to hold re-aligned mutations from parent-child
        self.aln_spike_mutations = []

    def initialize(self, node_dict, parent=None):
        """
        function to initialize node data from dictionary from auspice tree
        sets following attributes: name, branch_attrs, node_attrs, parent, root, leaf, children
        :param node_dict: node data from auspice tree
        :type node_dict: dict
        :param parent: parent node name
        :type parent: str
        """
        self.name = node_dict['name']
        self.branch_attrs = node_dict['branch_attrs']
        self.node_attrs = node_dict['node_attrs']
        self.parent = parent
        date = self.node_attrs['num_date']['value']
        self.date = get_date_from_num(date)
        if self.parent is None:
            self.root = True
        if has_children(node_dict):
            self.leaf = False
            for child in node_dict['children']:
                self.children.append(child['name'])
        else:
            self.leaf = True

    def get_path_from_dict(self, tree_node_dict):
        """
        function to set self.path from dictionary of key = nodename, value = node()
        if self.root, self.path is empty
        :param tree_node_dict: dictionary of key= nodename, value = node() for tree
        :type tree_node_dict: dict
        """
        root = self.root
        ancestor = self.parent
        while not root:
            # add current parent
            self.path.append(ancestor)
            # get node data for current parent
            parent_node = tree_node_dict[ancestor]
            ancestor = parent_node.parent
            root = parent_node.root

    def get_mutations_from_path(self, tree_node_dict):
        """
        function to get a list of mutations from all ndoes in self.path (ancestors)
        gets information from parent node class stored in tree_node_dict
        sets self.mutations
        :param tree_node_dict: dictionary of key=nodename, value=node() for tree
        :type tree_node_dict: dict
        """
        if 'S' in self.branch_attrs['mutations']:
            # mutations appear to be in ascending order.
            # reverse the list so that fartheset position is first for each node
            node_mutts = self.branch_attrs['mutations']['S']
            node_mutts.reverse()
            self.spike_mutations.extend(node_mutts)
            ordered_muts = sort_mut_list(node_mutts)
            self.node_spike_mutations = ordered_muts
        for node in self.path:
            branch_attrs = tree_node_dict[node].branch_attrs['mutations']
            if 'S' in branch_attrs:
                mutts = branch_attrs['S']
                # reverse these so farthest position is first for each ancestor node
                mutts.reverse()
                self.spike_mutations.extend(mutts)
        # reverse list so first mutations are from closer positions of farthers ancestors and last mutations are self
        self.spike_mutations.reverse()

    def get_spike_seq(self, reference_spike, ignore_deletion=True):
        """
         Function to get the spike sequence from list of mutations
        sets self.spike_seq
        :param reference_spike: reference spike protein sequence (needs to be a string)
        :type reference_spike: str
        :param ignore_deletion: bool flag to remove deletion tokens
        :type ignore_deletion: bool
        """
        self.spike_seq = mutate_sequence(reference_spike, self.spike_mutations, ignore_deletion=ignore_deletion)

    def realign_muts(self, tree_nodes):
        """
        function re-aligns node and parent sequence
        :param tree_nodes:
        :type tree_nodes:
        :return:
        :rtype:
        """
        if len(self.node_spike_mutations) > 0:
            parent_node = tree_nodes[self.parent]
            aln_muts = aln_parent_child_node(parent_node, self)
            self.aln_spike_mutations = sort_mut_list(aln_muts)


def read_auspice_file(nextstrain_path):
    """
    helper function to read in auspice tree file and return phylogenetic tree
    schema: https://github.com/nextstrain/augur/blob/master/augur/data/schema-export-v2.json
    :param nextstrain_path: path to auspice tree
    :type nextstrain_path: str
    :return: phylo tree dictionary
    :rtype: dict
    """
    f = open(nextstrain_path, )
    data = json.load(f)
    tree = data['tree']
    return tree


def get_tree_edges(nextstrain_path):
    """
    function to create a dataframe of edges (parent-child) from auspice tree
    :param nextstrain_path: path to auspice tree from nextstrain
    :type nextstrain_path: str
    :return: dataframe of edges
    :rtype: pandas.DataFrame
    """
    tree = read_auspice_file(nextstrain_path)
    tree_edges = tree_edge_list(tree)
    edge_df = pd.DataFrame(tree_edges)
    edge_df.columns = ['Parent', 'Child']
    return edge_df


def get_tree_data_dict(nextstrain_path, reference_path=None):
    """
    function to create a dictionary of key = node_id, value = Node from auspice output of nextstrain

    :param nextstrain_path: path to auspice tree from nextstrain
    :type nextstrain_path: str
    :param reference_path: path to reference spike protein record, used for finding spike sequences of nodes
    :type reference_path: str
    :return: dictionary of key=node_id, value = Node
    :rtype: dict
    """
    tree = read_auspice_file(nextstrain_path)
    tree_nodes = tree_node_data(tree)
    for node_id, node in tqdm(tree_nodes.items(), desc='getting node paths'):
        node.get_path_from_dict(tree_nodes)
    # have to finish getting path before mutations
    for node_id, node in tqdm(tree_nodes.items(), desc='getting node mutations'):
        node.get_mutations_from_path(tree_nodes)
    if reference_path is None:
        ref_path = 'data/ref_fastas/reference_spike.fasta'
    else:
        ref_path = reference_path
    ref_spike_record = list(SeqIO.parse(ref_path, 'fasta'))[0]
    ref_seq = str(ref_spike_record.seq)
    for node_id, node in tqdm(tree_nodes.items(), desc='getting node spike'):
        node.get_spike_seq(reference_spike=ref_seq)
    for node_id, node in tqdm(tree_nodes.items(), desc='realigning node mutations'):
        node.realign_muts(tree_nodes)
    return tree_nodes


def mutation_summary(tree_nodes):
    """
    function to get all mutations from leaf nodes in tree_nodes dictionary.
    summarizes mutations: first date seen, n_times seen (in leafs), n_gen (from root)
    :param tree_nodes: tree node data dictionary, key=node_id, value=Node
    :type tree_nodes: dict
    :return: dataframe with mutation summary
    :rtype: pandas.Dataframe
    """
    mutt_dict = {}
    for node_id, node in tree_nodes.items():
        if node.leaf:
            mutts = node.spike_mutations
            for mutt in mutts:
                if mutt not in mutt_dict:
                    first_date = node.date
                    n_to_root = len(node.path)
                    mutt_dict[mutt] = {
                        'first_date': first_date,
                        'n_times': 1,
                        'n_gen': n_to_root
                    }
                else:
                    meta = mutt_dict[mutt]
                    date = node.date
                    if date < meta['first_date']:
                        meta['first_date'] = date
                        meta['n_gen'] = len(node.path)
                    meta['n_times'] = meta['n_times'] + 1
    df = pd.DataFrame.from_dict(mutt_dict).T.reset_index()
    df.columns = ['mutation', 'first_date', 'n_times', 'n_gen']
    df[['wt', 'pos', 'mut']] = df.apply(lambda x: pullout_pos_mut_df(x), axis=1, result_type='expand')
    return df


def analyze_auspice_tree(tree_version):
    """
    function created edge_df, tree_nodes, mutation summary df from auspice tree tree
    saves all in subfolder under processed based on tree version
    returns objects if wanted
    :param tree_version: number indicating the round of sampling from nextstrain
    :type tree_version: int
    :return: edge_df, tree_nodes, mutt_df
    :rtype: tuple
    """
    root = 'data/processed'
    folder = root + '/ncbi_tree_v{}'.format(tree_version)
    file = 'ncov_tree_v{}.json'.format(tree_version)
    path = folder +'/'+ file
    edge_df = get_tree_edges(path)
    edge_df.to_pickle(folder+'/tree_edges_v{}.pkl'.format(tree_version))
    n_edges = len(edge_df)
    tree_nodes = get_tree_data_dict(path)
    with open(folder+'/tree_nodes_v{}.pkl'.format(tree_version), 'wb') as f:
        pickle.dump(tree_nodes, f)
    n_nodes = len(tree_nodes)
    n_leaf = 0
    for node_id, node in tree_nodes.items():
        if node.leaf:
            n_leaf = n_leaf + 1
    mutt_df = mutation_summary(tree_nodes)
    mutt_df.to_pickle(folder+'/leaf_mutts_tree_v{}.pkl'.format(tree_version))
    message = "Number Nodes:{} \nNumber Leaf:{} \nNumber Edges:{}".format(n_nodes, n_leaf, n_edges)
    with open(folder + '/tree_v{}_info.txt'.format(tree_version), 'w') as f:
        f.write(message)
    return edge_df, tree_nodes, mutt_df


def get_training_seq(tree_version, cut_off="2022-1-1", unique=True):
    """
    function to get a list of training sequences from a given tree (saved in subfolder under data/processed)
    specify a cut off and gets leaf sequences that happened prior to the cut-off
    if unique then return only unique sequences
    :param unique: if true return unique sequences only
    :type unique: bool
    :param tree_version: indicates which round sampling tree to grab
    :type tree_version: int
    :param cut_off: cut off date of nodes to be included in training set, format like yyyy-m-d
    :type cut_off: str
    :return: list of training sequences
    :rtype: list
    """
    root = 'data/processed'
    folder = root + '/ncbi_tree_v{}'.format(tree_version)
    nodes_path = folder+'/tree_nodes_v{}.pkl'.format(tree_version)
    with open(nodes_path, 'rb') as f:
        tree_nodes = pickle.load(f)
    cut_off_datetime = datetime.strptime(cut_off, '%Y-%m-%d')
    training_seq = []
    for node_id, node in tree_nodes.items():
        if node.leaf:
            date = node.date
            if date < cut_off_datetime:
                seq = node.spike_seq
                training_seq.append(seq)
    n_seq = len(training_seq)
    unique_seq = list(set(training_seq))
    n_unique = len(unique_seq)
    message = "Cut-off:{} \nNumber Training Seq:{} \nNumber Unique Training Seq:{}".format(cut_off, n_seq, n_unique)
    with open(folder + '/tree_v{}_train_seq_info.txt'.format(tree_version), 'w') as f:
        f.write(message)
    with open(folder+'/tree_train_full_v{}.pkl'.format(tree_version), 'wb') as f:
        pickle.dump(training_seq, f)
    with open(folder+'/tree_train_unique_v{}.pkl'.format(tree_version), 'wb') as f:
        pickle.dump(unique_seq, f)
    if unique:
        return unique_seq
    else:
        return training_seq


if __name__ == "__main__":
    versions = [1]
    for version_num in versions:
        print('Version: {}'.format(version_num))
        print('processing auspice tree')
        _ = analyze_auspice_tree(version_num)
        _ = get_training_seq(version_num)

    print('done')
