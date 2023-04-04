import os
import pickle
import pandas as pd
from datetime import datetime
from PhyloTreeParsers import Node
from BioTransLanguageModel import *
from MutationRankingResults import mutation_rank_results, combine_model_tree, seq_mutation_dict_results
from tqdm import tqdm
from biotransformers import BioTransformers
import torch
import ray
from MutationHelpers import *
import argparse
from DataHelpers import rename_multicol_df, check_directory
import uuid


def parent_child_candidates(tree_nodes, cut_off='2022-1-1'):
    """
    function to get unique parent-child sequences from tree_nodes dict
    :param tree_nodes: dictionary of key = node_id, value = Node
    :type tree_nodes: dict
    :param cut_off: cut off date, get nodes that are after this date
    :type cut_off: str
    :return: dictionary of parent-child candidates, key = parentid, value = dict ('children' = list of child_ids)
    :rtype: dict
    """
    cut_off_datetime = datetime.strptime(cut_off, '%Y-%m-%d')
    post_cutoff = pd.DataFrame()
    i = 0
    for node_id, node in tree_nodes.items():
        if node.date > cut_off_datetime and not node.leaf:
            parent_id = node_id
            parent_seq = node.spike_seq
            for child_id in node.children:
                child_node = tree_nodes[child_id]
                muts = child_node.node_spike_mutations
                muts = [x for x in muts if x.find('-') == -1]
                if len(muts) > 0:
                    child_seq = child_node.spike_seq
                    row = pd.DataFrame({
                        'parent_id': parent_id,
                        'child_id': child_id,
                        'parent_seq': parent_seq,
                        'child_seq': child_seq
                    }, index=[i])
                    i = i + 1
                    post_cutoff = post_cutoff.append(row)
    post_cutoff = post_cutoff.drop_duplicates(subset=['parent_seq', 'child_seq'], keep='first')
    post_cutoff = post_cutoff[['parent_id', 'child_id']]
    candidates = post_cutoff.groupby('parent_id')['child_id'].apply(list).to_dict()
    return candidates


def candidates_by_mutations(tree_nodes, sig_muts):
    """

    :param tree_nodes:
    :type tree_nodes:
    :param sig_muts: list of mutations that determine the candidate child nodes
    :type sig_muts: list
    :return:
    :rtype:
    """
    # the WT in sig_muts list is in reference to reference (tree root)
    # but the WT in node.node_spike_mutations might be different, but should be same mutation
    sig_muts_no_wt = [x[1:] for x in sig_muts]
    candidate_parents = []
    for node_id, node in tree_nodes.items():
        muts = node.node_spike_mutations
        muts_no_wt = [x[1:] for x in muts]
        overlap = [x for x in muts_no_wt if x in sig_muts_no_wt]
        if len(overlap) > 0:
            candidate_parents.append(node.parent)
    candidate_parents = list(set(candidate_parents))
    post_cutoff = pd.DataFrame()
    i = 0
    for node_id, node in tree_nodes.items():
        if node_id in candidate_parents:
            parent_id = node_id
            parent_seq = node.spike_seq
            for child_id in node.children:
                child_node = tree_nodes[child_id]
                muts = child_node.node_spike_mutations
                muts = [x for x in muts if x.find('-') == -1]
                muts_no_wt = [x[1:] for x in muts]
                muts = [x for x in muts_no_wt if x in sig_muts_no_wt]
                if len(muts) > 0:
                    child_seq = child_node.spike_seq
                    row = pd.DataFrame({
                        'parent_id': parent_id,
                        'child_id': child_id,
                        'parent_seq': parent_seq,
                        'child_seq': child_seq
                    }, index=[i])
                    i = i + 1
                    post_cutoff = post_cutoff.append(row)
    post_cutoff = post_cutoff.drop_duplicates(subset=['parent_seq', 'child_seq'], keep='first')
    post_cutoff = post_cutoff[['parent_id', 'child_id']]
    candidates = post_cutoff.groupby('parent_id')['child_id'].apply(list).to_dict()
    return candidates


def realign_candidates(candidates, tree_nodes, sig_muts, train_seq=None):
    """
    function to take candidates dict and re-align parent-child
    if re-alignment mutations are correct position between parent-child, then add to final parent-child dictionary
    if list of train_seq then add a check into meta data if sequence is in training
    :param sig_muts: list of mutations from reference experiment that are significant
    :type sig_muts: list
    :param train_seq: if provided, adds to child meta data if the child sequence is in training set
    :type train_seq: list
    :param candidates: dictionary of key = parent, value = list of children
    :type candidates: dict
    :param tree_nodes: dictionary of key = node_id, value = Node
    :type tree_nodes: dict
    :return: dictionary of key = parent_id, value = list of children
    :rtype:
    """
    reference_spike = load_ref_spike()
    reference_spike = str(reference_spike.seq)
    sig_muts_no_wt = [x[1:] for x in sig_muts]
    parent_child = {}
    for parent_id, children in candidates.items():
        parent_node = tree_nodes[parent_id]
        parent_seq = parent_node.spike_seq
        for child_id in children:
            child_node = tree_nodes[child_id]
            child_seq = child_node.spike_seq
            muts = child_node.node_spike_mutations
            muts = [x for x in muts if x.find('-') == -1]
            node_muts = [x for x in muts if x[1:] in sig_muts_no_wt]
            muts_from_ref = node_muts_from_ref(child_node.spike_mutations)
            map_ref_pos_to_parent_pos = ref_seq_pos_map(reference_spike, mutations=parent_node.spike_mutations)

            corrected_mut_map = {}
            for mut in node_muts:
                wt, pos, mutted = pullout_pos_mut(mut)
                parent_pos = map_ref_pos_to_parent_pos[pos]
                if parent_seq[parent_pos] == wt:
                    corrected_mut = "{}{}{}".format(wt, parent_pos + 1, mutted)
                    ref_mut = [x for x in muts_from_ref if x[1:] == mut[1:]][0]
                    #corrected_mut_map[corrected_mut] = mut
                    corrected_mut_map[corrected_mut] = ref_mut
            if len(corrected_mut_map) > 0:
                corrected_muts = list(corrected_mut_map.keys())
                child_dict = {
                    'child_id': child_id,
                    'node_muts': node_muts,
                    #'aln_muts': aln_muts,
                    'corrected_muts': corrected_muts,
                    'corrected_mut_map': corrected_mut_map
                }
                if train_seq is not None:
                    child_dict['child_in_train'] = child_seq in train_seq
                if parent_id not in parent_child:
                    parent_child[parent_id] = []
                parent_child[parent_id].append(child_dict)
    return parent_child


def biotrans_probabilities(seq, bio_trans, forward_mode=True):
    """

    :param bio_trans:
    :type bio_trans:
    :param seq:
    :type seq:
    :param forward_mode:
    :type forward_mode:
    :return:
    :rtype:
    """
    if forward_mode:
        probabilities = bio_trans.compute_probabilities([seq], pass_mode='forward')
    else:
        probabilities = bio_trans.compute_probabilities([seq], pass_mode='masked')
    probabilities = probabilities[0]
    return probabilities


def mutate_parent(parent_seq, bio_trans, forward_mode=True, sig_muts=None):
    """

    :param forward_mode: if true, use forward mode, else masked mode, default is true
    :type forward_mode: bool
    :param parent_seq:
    :type parent_seq:
    :param bio_trans:
    :type bio_trans:
    :param sig_muts:
    :type sig_muts:
    :return:
    :rtype:
    """
    seq_muts = mutate_seq_insilico(parent_seq, significant_mutations=sig_muts)
    probabilities = biotrans_probabilities(parent_seq, bio_trans, forward_mode=forward_mode)
    seq_muts = get_mutation_probabilities(parent_seq, probabilities, seqs_mutated=seq_muts)
    return seq_muts


def mark_significant(seq_muts, sig_mut_lst):
    """
    function to take seq_mut dict (key = mutated sequence, value = meta data with probability and mutation string)
    and re-mark what mutations are significant
    :param seq_muts: dict (key = mutated sequence, value = meta data with probability and mutation string)
    :type seq_muts: dict
    :param sig_mut_lst: list of significant mutations formatted as [wt][pos+1][mut]
    :type sig_mut_lst: lst
    :return: dictionary with recalculated significant in meta
    :rtype: dict
    """
    for seq, meta in seq_muts.items():
        meta['significant'] = meta['mutation'] in sig_mut_lst
    return seq_muts


def seq_mutated_rank_results(seq_muts, legend_title, **kwargs):
    """

    :param legend_title:
    :type legend_title:
    :param seq_muts:
    :type seq_muts:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    seqs = list(seq_muts.keys())
    prob_list = []
    gt_idx = []
    for seq in seqs:
        prob = seq_muts[seq]['prob']
        prob_list.append(prob)
        gt = seq_muts[seq]['significant']
        gt_idx.append(gt)
    prob_list = np.array(prob_list)
    gt_idx = np.array(gt_idx)
    results = mutation_rank_results(prob_list, gt_idx, title=legend_title, **kwargs)
    result_keys = ['auc', 'mean_rank', 'max_rank', 'min_rank', 'n_gt']
    results = {k: v for k, v in results.items() if k in result_keys}
    return results


def freq_str(row, mut_str_col, mut_dict):
    """
    function to take a list of mutations and return their values from the mut_dict
    :param row:
    :type row:
    :param mut_str_col:
    :type mut_str_col:
    :param mut_dict:
    :type mut_dict:
    :return:
    :rtype:
    """
    mut_str = row[mut_str_col]
    muts = mut_str.split(';')
    muts = [x.strip() for x in muts]
    dict_values = []
    for m in muts:
        wt, pos, alt = pullout_pos_mut(m)
        n = mut_dict[(pos, alt)]
        dict_values.append(n)
    return "; ".join(dict_values)


def n_unique_muts(grouped):
    """
    grouper agg function
    returns number of unique mutations from column, some values in column are separated by ';"
    :param grouped: grouped column
    :type grouped: pandas.Series
    :return: number unique mutations
    :rtype: int
    """
    muts = '; '.join(grouped)
    n_unique = len(set([x.strip() for x in muts.split(';')]))
    return n_unique


class BioTransExpSettings(object):
    """

    """
    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, model_folder=None, l1_change=False,
                 cut_off='2022-1-1'):
        self.cut_off = cut_off
        self.tree_version = tree_version
        self.finetuned = finetuned
        self.forward_mode = forward_mode
        self.data_folder = data_folder
        self.l1_change = l1_change
        self.seq_change_path = None
        self.seq_prob_path = None
        self.exp_folder = self.data_folder + "/exp_settings"
        if model_folder is not None:
            self.model_folder = model_folder
        else:
            file = open(self.exp_folder + "/model_folder.txt", "r")
            self.model_folder = file.read()
            file.close()
        self.tree_nodes = None
        self.model_path = None
        self.load_data()

    def load_data(self):
        tree_folder = 'data/processed/ncbi_tree_v{}'.format(self.tree_version)
        # load tree_nodes
        nodes_path = tree_folder + '/tree_nodes_v{}.pkl'.format(self.tree_version)
        with open(nodes_path, 'rb') as file:
            self.tree_nodes = pickle.load(file)
        if self.finetuned:
            self.model_path = last_biotrans_ft_path(self.model_folder)
            print(self.model_path)
        if self.forward_mode:
            mode_str = 'forward'
        else:
            mode_str = 'masked'
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        if self.l1_change:
            norm = 'l1'
        else:
            norm = 'l2'
        self.seq_change_path = self.data_folder + '/tree_v{}_seq_{}_change_{}.pkl'.format(self.tree_version, norm, train_str)
        self.seq_prob_path = self.data_folder + '/tree_v{}_seq_{}_prob_{}.pkl'.format(self.tree_version, mode_str, train_str)


class MutateParentChild(BioTransExpSettings):
    """

    """

    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, model_folder=None, l1_change=True):
        super().__init__(tree_version, data_folder, finetuned, forward_mode, model_folder, l1_change, cut_off='2022-1-1')
        self.train_seq = None
        self.mutation_data = None
        self.new_mutations = None
        self.ref_results = None
        self.ref_exp_muts = []
        self.new_muts = []
        self.old_muts = []
        self.parent_child = {}
        self.seq_probabilities = {}
        self.seq_change = {}
        # dict to hold counter of aln_mut - count ref_mut
        #xself.aln_mapped_cnt = {}
        self.results = pd.DataFrame()
        self.results_summary = pd.DataFrame()
        self.prep_data()
        self.seqs = []

    def prep_data(self):
        print('loading tree data')
        self.load_exp_data()
        print('prepping parent child')
        self.prep_parent_child()

    def run_experiment(self, excel=True, save_class=False, include_change=True, threshold_max=3, save_name=None,
                       load_previous=False, subset_parentids=None, **kwargs):
        """
        kwargs include: prob_seq_batchsize (chunksize for sequences in probabilities); prob_batchsize (batchsize for
        biotrans compute probabiltiies); seq_batchsize (batches for embdding); embedding_batchsize
        (batchsize for biotrans embedding)

        :param threshold_max: max threshold for mutate_parents, default is 3
        :type threshold_max: int
        :param include_change: if true, calculate embedding semantic change
        :type include_change: bool
        :param excel: if true, save data in excel format
        :type excel: bool
        :param save_class: if true, pickle the whole class
        :type save_class: bool
        :param kwargs: dictionary of params for sequence_language_model_values
        :type kwargs: dict
        :return:
        :rtype:
        """
        if subset_parentids is not None:
            self.parent_child = {k:self.parent_child[k] for k in subset_parentids}
        self.sequence_language_model_values(include_change=include_change, **kwargs)
        if load_previous:
            self.load_exp_results(save_name=save_name, excel=excel)
        else:
            self.mutate_parents(include_change=include_change, threshold_max=threshold_max)
            self.mut_summary()
            self.save_data(excel=excel, save_class=save_class, save_name=save_name)

    def load_exp_data(self):
        # load leaf mutt dict
        tree_df_path = self.exp_folder + '/mutation_data.pkl'
        print(tree_df_path)
        self.mutation_data = pd.read_pickle(tree_df_path)
        self.new_mutations = self.mutation_data[(self.mutation_data['first_date'] >= self.cut_off)]
        self.new_mutations['pos'] = self.new_mutations['pos'].astype(int)
        self.new_muts = self.new_mutations['mutation'].values.tolist()
        self.old_muts = self.mutation_data[~(self.mutation_data.mutation.isin(self.new_muts))]['mutation'].values.tolist()
        # load reference results
        ref_path = self.exp_folder + '/pbert_ft_ref.pkl'
        ref = pd.read_pickle(ref_path)
        tree_df = self.new_mutations[~(self.new_mutations['mutation'].str.contains('-'))]
        self.ref_results = combine_model_tree(ref, tree_df)
        self.ref_exp_muts = self.ref_results[(self.ref_results['tree_mut_all'])]['mutation'].values.tolist()
        # load train_seq
        train_seq_path = self.exp_folder + '/train_seq.pkl'
        with open(train_seq_path, 'rb') as file:
            self.train_seq = pickle.load(file)

    def prep_parent_child(self):
        self.parent_child = candidates_by_mutations(self.tree_nodes, self.ref_exp_muts)
        self.parent_child = realign_candidates(self.parent_child, self.tree_nodes, self.ref_exp_muts, self.train_seq)
        # get the counter maps
        #all_mut_cnt = self.mutation_data.set_index(['pos','mut']).n_times.to_dict()
        all_mut_cnt = self.mutation_data.set_index(['mutation']).n_times.to_dict()
        for parent_id, children in self.parent_child.items():
            for child_meta in children:
                mapped = child_meta['corrected_mut_map']
                # mapped_cnt is key = significant mutation, value = frequency of reference mapped mutation
                child_meta['mapped_cnt'] = {}
                for aln, aln_mapped in mapped.items():
                    #wt, pos, alt = pullout_pos_mut(aln_mapped)
                    #cnt = all_mut_cnt[(pos, alt)]
                    cnt = all_mut_cnt[aln_mapped]
                    child_meta['mapped_cnt'][aln] = cnt
                    #self.aln_mapped_cnt[aln] = cnt
        # sub records in for testing
        # keys = list(self.parent_child.keys())[0:10]
        # self.parent_child = {key: self.parent_child[key] for key in keys}

    def get_sequences(self):
        for node_id, node in self.tree_nodes.items():
            if node.root:
                self.seqs.append(node.spike_seq)
        for k, v in self.parent_child.items():
            seq = self.tree_nodes[k].spike_seq
            self.seqs.append(seq)
        self.seqs = list(set(self.seqs))
        print('Total Unique Parent Sequences: {}'.format(len(self.seqs)))

    def sequence_language_model_values(self, list_seq_batchsize=15, prob_batchsize=20, seq_batchsize=400,
                                       embedding_batchsize=40, include_change=True, run_chunked_change=False, combine=False):
        self.get_sequences()
        print('checking saved probabilities')
        seqs_for_proba, self.seq_probabilities = find_previous_saved(self.seqs, self.seq_prob_path)
        if include_change:
            print('checking saved embedding change')
            seqs_for_change, self.seq_change = find_previous_saved(self.seqs, self.seq_change_path)
        else:
            seqs_for_change = []
        if len(seqs_for_proba) > 0 or len(seqs_for_change) > 0:
            bio_trans = load_biotrans(model_path=self.model_path)
            if len(seqs_for_proba) > 0:
                print('computing probabilities for {} sequences'.format(len(seqs_for_proba)))
                self.seq_probabilities = compute_probabilities(bio_trans=bio_trans, seqs=seqs_for_proba,
                                                               batchsize=prob_batchsize, chunksize=list_seq_batchsize,
                                                               forward_mode=self.forward_mode,
                                                               prior_seq_probabilities=self.seq_probabilities,
                                                               save_path=self.seq_prob_path)

                print('Saving {} sequence probabilities'.format(len(self.seq_probabilities)))
                with open(self.seq_prob_path, 'wb') as a:
                    pickle.dump(self.seq_probabilities, a)
            if len(seqs_for_change) > 0:
                if run_chunked_change:
                    self.parent_semantic_change_batched(bio_trans=bio_trans, seq_batchsize=seq_batchsize,
                                                        embedding_batchsize=embedding_batchsize,
                                                        list_seq_batchsize=list_seq_batchsize, combine=combine)
                else:
                    print('computing semantic change for {} sequences'.format(len(seqs_for_change)))
                    self.seq_change = embedding_change_batchs(seqs=seqs_for_change, bio_trans=bio_trans,
                                                              seq_batchsize=seq_batchsize,
                                                              embedding_batchsize=embedding_batchsize,
                                                              seq_change=self.seq_change,
                                                              save_path=self.seq_change_path,
                                                              chunksize=list_seq_batchsize, l1_norm=self.l1_change)
                    print('Saving {} sequence semantic change'.format(len(self.seq_change)))
                    with open(self.seq_change_path, 'wb') as a:
                        pickle.dump(self.seq_change, a)

    def parent_semantic_change_batched(self, bio_trans, seq_batchsize, embedding_batchsize, list_seq_batchsize,
                                       combine=False):
        str_id = uuid.uuid1()
        if self.l1_change:
            change_folder = self.data_folder + "/l1_changes"
        else:
            change_folder = self.data_folder + '/l2_changes'
        check_directory(change_folder)
        suffix = self.seq_change_path.split('/')[-1]
        change_save_files = []
        #if os.path.isfile(self.seq_change_path):
        #    change_save_files.append(self.seq_change_path)
        for file in os.listdir(change_folder):
            if file.endswith(suffix):
                change_save_files.append(change_folder+'/'+file)
        current_save_path = change_folder + '/{}_{}'.format(str_id, suffix)

        seq_dict = self.combine_seq_dicts(file_list = change_save_files)
        self.seq_change.update(seq_dict)
        seqs_for_change = []
        for seq in self.seqs:
            if seq not in self.seq_change:
                seqs_for_change.append(seq)
        if len(seqs_for_change) > 0:
            print('computing semantic change for {} sequences'.format(len(seqs_for_change)))
            self.seq_change = embedding_change_batchs(seqs=seqs_for_change, bio_trans=bio_trans,
                                                    seq_batchsize=seq_batchsize,
                                                    embedding_batchsize=embedding_batchsize,
                                                    seq_change=self.seq_change,
                                                    save_path=current_save_path,
                                                    chunksize=list_seq_batchsize, l1_norm=self.l1_change)
            print('Saving {} sequence semantic change'.format(len(self.seq_change)))
            with open(current_save_path, 'wb') as a:
                pickle.dump(self.seq_change, a)
        if combine:
            if os.path.isfile(self.seq_change_path):
                change_save_files.append(self.seq_change_path)
            seq_dict = self.combine_seq_dicts(file_list=change_save_files)
            self.seq_change.update(seq_dict)
            print('Saving {} combined sequence semantic change'.format(len(self.seq_change)))
            with open(self.seq_change_path, 'wb') as a:
                pickle.dump(self.seq_change, a)


    def combine_seq_dicts(self, file_list):
        seq_dict = {}
        for file in file_list:
            with open(file, 'rb') as f:
                values = pickle.load(f)
            seq_dict.update(values)
        return seq_dict




    def parent_probabilities(self):
        if self.forward_mode:
            mode_str = 'forward'
        else:
            mode_str = 'masked'
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        seq_prob_path = self.data_folder + '/tree_v{}_seq_{}_prob_{}.pkl'.format(self.tree_version, mode_str, train_str)
        if os.path.isfile(seq_prob_path):
            print('previous saved probabilities found')
            print(seq_prob_path)
            with open(seq_prob_path, 'rb') as file:
                self.seq_probabilities = pickle.load(file)
        else:
            print('no previous saved probabilities')
        seqs = []
        for node_id, node in self.tree_nodes.items():
            if node.root and node.spike_seq not in self.seq_probabilities:
                seqs.append(node.spike_seq)
        for k, v in self.parent_child.items():
            seq = self.tree_nodes[k].spike_seq
            if seq not in self.seq_probabilities:
                seqs.append(seq)
        seqs = list(set(seqs))
        print("computing probabilities for {} sequences.".format(len(seqs)))
        if len(seqs) > 0:
            n_gpu = torch.cuda.device_count()
            ray.init()
            bio_trans = BioTransformers(backend="protbert", num_gpus=n_gpu)
            if self.model_path is not None:
                bio_trans.load_model(self.model_path)
            for seq in seqs:
                probabilities = biotrans_probabilities(seq, bio_trans, forward_mode=self.forward_mode)
                self.seq_probabilities[seq] = probabilities
        with open(seq_prob_path, 'wb') as file:
            pickle.dump(self.seq_probabilities, file)

    def mutate_parents(self, threshold_max=3, include_change=True):
        data_list = []
        for parent_id, children in tqdm(self.parent_child.items(), desc='Unique Parent Children'):
            parent_seq = self.tree_nodes[parent_id].spike_seq
            parent_in_train = parent_seq in self.train_seq
            probabilities = self.seq_probabilities[parent_seq]
            if include_change:
                changes = self.seq_change[parent_seq]
            else:
                changes = None
            for child_meta in children:
                child_id = child_meta['child_id']
                sig_muts = child_meta['corrected_muts']
                result_meta = {'result_type': 'All',
                               'threshold': 0,
                               'parent_id': parent_id,
                               'parent_in_train': parent_in_train,
                               'child_id': child_id,
                               'child_in_train': child_meta['child_in_train'],
                               'muts_str': "; ".join(sig_muts),
                               'tree_muts_str': '; '.join(child_meta['node_muts']),
                               'n_gt': len(sig_muts)
                               }
                # run with all muts
                seq_mutations = get_seq_mutation_dict(parent_seq, probabilities, changes, sig_muts)
                results = seq_mutation_dict_results(seq_mutations)
                result_meta.update(results)
                data_list.append(result_meta.copy())

                # run with only new muts
                mut_map = child_meta['corrected_mut_map']
                sig_muts = [x for x in sig_muts if x in mut_map and mut_map[x] in self.ref_exp_muts]
                #key = sig mut, value = frequency of reference mapped mutation
                mapped_cnt = child_meta['mapped_cnt']
                freqs = [mapped_cnt[x] for x in sig_muts]
                freqs = [x - 1 for x in freqs]
                freqs = list(set(freqs))
                freqs = sorted(freqs)
                for threshold in freqs:
                    new_sig_muts = [x for x in sig_muts if x in mapped_cnt and mapped_cnt[x] > threshold]
                    new_tree_muts = [mut_map[x] for x in new_sig_muts]
                    if len(new_sig_muts) > 0:
                        result_meta['result_type'] = 'New'
                        result_meta['threshold'] = threshold
                        result_meta['n_gt'] = len(new_sig_muts)
                        seq_mutations = mark_significant(seq_mutations, new_sig_muts)
                        results = seq_mutation_dict_results(seq_mutations)
                        result_meta.update(results)
                        result_meta['muts_str'] = "; ".join(new_sig_muts)
                        result_meta['tree_muts_str'] = "; ".join(new_tree_muts)
                        data_list.append(result_meta.copy())
        self.results = pd.DataFrame(data_list)

    def mut_summary(self):
        self.mutation_data['n_times_str'] = self.mutation_data.n_times.astype('str')
        freq_dict = self.mutation_data.set_index(['pos', 'mut']).n_times_str.to_dict()
        self.mutation_data['first_date_str'] = self.mutation_data['first_date'].dt.strftime('%Y%m%d')
        date_dict = self.mutation_data.set_index(['pos', 'mut']).first_date_str.to_dict()
        mut_name_dict = self.mutation_data.set_index(['pos', 'mut']).mutation.to_dict()
        self.results['ref_muts_str'] = self.results.apply(lambda x: freq_str(x, 'tree_muts_str', mut_name_dict), axis=1)
        self.results['muts_freq'] = self.results.apply(lambda x: freq_str(x, 'tree_muts_str', freq_dict), axis=1)
        self.results['muts_date'] = self.results.apply(lambda x: freq_str(x, 'tree_muts_str', date_dict), axis=1)
        df = self.results[~(self.results['child_in_train'])]

        groupby_col = ['result_type', 'threshold', 'ref_muts_str', 'n_gt', 'cscs_auc', 'change_auc', 'prob_auc']
        groupby_col = [x for x in groupby_col if x in df]
        df = df.groupby(groupby_col)['parent_id'].count().reset_index()
        agg_cols = ['cscs_auc', 'change_auc', 'prob_auc']
        agg_dict = {'parent_id': 'sum'}
        agg_dict.update({x: 'mean' for x in agg_cols if x in df})
        groupby_col = [x for x in groupby_col if x not in agg_dict]
        df = df.groupby(groupby_col).agg(agg_dict).reset_index()
        df['threshold'] = df['threshold'].astype(int)
        thresholds = sorted(list(set(df['threshold'])))
        agg_dict = {'ref_muts_str': n_unique_muts, 'parent_id': 'sum', 'n_gt': 'mean',
                    'cscs_auc': ['mean', 'std'], 'change_auc': ['mean', 'std'], 'prob_auc': ['mean', 'std']
                    }
        agg_dict = {k: agg_dict[k] for k in list(agg_dict.keys()) if k in df}
        self.results_summary = pd.DataFrame()
        for threshold in thresholds:
            df1 = df[(df['threshold'] >= threshold)].groupby('result_type').agg(agg_dict).reset_index()
            df1 = rename_multicol_df(df1)
            df1.insert(1, 'threshold', threshold)
            self.results_summary = self.results_summary.append(df1)
        print(self.results_summary)

    def get_savename(self, save_name=None):
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        if self.forward_mode:
            mode_str = 'forward'
        else:
            mode_str = 'masked'
        if self.l1_change:
            norm = 'l1'
        else:
            norm = 'l2'
        if save_name is not None:
            base_name = save_name
        else:
            base_name = 'pc_tree_v{}_{}_{}_{}_mode'.format(self.tree_version, train_str, norm, mode_str)
        return base_name

    def save_data(self, excel=True, save_class=False, save_name=None):
        save_folder = self.data_folder + "/results"
        base_name = self.get_savename(save_name=save_name)
        check_directory(save_folder)
        if excel:
            file_name = base_name + '.xlsx'
            summary_file_name = base_name + '_summary.xlsx'
            self.results.to_excel(save_folder + '/' + file_name)
            self.results_summary.to_excel(save_folder + '/' + summary_file_name)
        else:
            file_name = base_name + '.pkl'
            summary_file_name = base_name + "_summary.pkl"
            self.results.to_pickle(save_folder + '/' + file_name)
            self.results_summary.to_pickle(save_folder + '/' + summary_file_name)
        if save_class:
            with open(base_name + '_MutateParentChild.pkl', 'wb') as file:
                pickle.dump(self, file)

    def load_exp_results(self, save_name=None, excel=False):
        save_folder = self.data_folder + "/results"
        base_name = self.get_savename(save_name=save_name)
        if excel:
            file_name = base_name + '.xlsx'
            summary_file_name = base_name + '_summary.xlsx'
            try:
                self.results = pd.read_excel(save_folder + '/' + file_name)
                self.results_summary = pd.read_excel(save_folder + '/' + summary_file_name)
            except FileNotFoundError:
                print("Designated Savefile doesn't exist!")
        else:
            file_name = base_name + '.pkl'
            summary_file_name = base_name + "_summary.pkl"
            try:
                self.results = pd.read_pickle(save_folder + '/' + file_name)
                self.results_summary = pd.read_pickle(save_folder + '/' + summary_file_name)
            except FileNotFoundError:
                print("Designated Savefile doesn't exist!")



def parse_args():
    parser = argparse.ArgumentParser(description='ParentChild Mutate experiment')
    parser.add_argument('--tree_version', type=int, help='Tree Version for models')
    parser.add_argument('--finetuned', action='store_true',
                        help='to use fine tuned biotransformer model')
    parser.add_argument('--masked_mode', action='store_false',
                        help='if provided, experiment will use masked mode')
    parser.add_argument('--l2_norm', action='store_false',
                        help='if provided, experiment will use l2 norm')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
    parser.add_argument('--include_change', action='store_true',
                        help='include change in experiment')
    parser.add_argument('--change_batched', action='store_true',
                        help='batch save file changes')
    parser.add_argument('--combine', action='store_true',
                        help='combine batch save file changes')
    parser.add_argument('--list_seq_batchsize', type=int, default=15,
                        help='number of sequences to calculate embedding change/probabilities at a time')
    parser.add_argument('--prob_batchsize', type=int, default=20,
                        help='token batchsize for calculating prob')
    parser.add_argument('--seq_batchsize', type=int, default=500,
                        help='sequence list batch for embedding')
    parser.add_argument('--embedding_batchsize', type=int, default=50,
                        help='mini sequence batchsize for embedding')
    parser.add_argument('--sub_parentid_path', type=str, default=None,
                        help='if provided, load a list of parent_ids to subset the experiment to')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    parent_child_exp = MutateParentChild(tree_version=args.tree_version,
                                         finetuned=args.finetuned,
                                         forward_mode=args.masked_mode,
                                         data_folder=args.data_folder,
                                         l1_change = args.l2_norm)
    print("masked mode arg: {}".format(args.masked_mode))
    print("Finetuned is: {}".format(parent_child_exp.finetuned))
    print("Forward Mode is: {}".format(parent_child_exp.forward_mode))
    print("L1 change is: {}".format(parent_child_exp.l1_change))

    if args.sub_parentid_path is not None:
        with open(args.sub_parentid_path, 'rb') as f:
            sub_parentids = pickle.load(f)
    else:
        sub_parentids = None



    #params = {'list_seq_batchsize': 15,
    #          'prob_batchsize': 20,
    #          'seq_batchsize': 400,
    #          'embedding_batchsize': 40}
    params = {'list_seq_batchsize': args.list_seq_batchsize,
              'prob_batchsize': args.prob_batchsize,
              'seq_batchsize': args.seq_batchsize,
              'embedding_batchsize': args.embedding_batchsize,
              'run_chunked_change': args.change_batched,
              'combine': args.combine}

    print(params)
    parent_child_exp.run_experiment(include_change=args.include_change, excel=False,
                                    subset_parentids=sub_parentids, **params)
