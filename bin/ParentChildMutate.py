import hashlib
from PhyloTreeParsers import Node
from BioTransLanguageModel import *
from MutationRankingResults import seq_mutation_data_results, results_over_thresholds
from tqdm import tqdm
from MutationHelpers import *
import argparse
from DataHelpers import check_directory, combine_seq_dicts, folder_file_list
import uuid
import json


def candidates_by_mutations(tree_nodes, sig_muts, drop_by_muts=False):
    """
    Preps parent-child candidates for mutation by identifing childing with mutations in sig_muts
    :param drop_by_muts: if True, focus on unique parent sequence & significant mutation
    :type drop_by_muts: bool
    :param tree_nodes: tree nodes to search for
    :type tree_nodes: dict
    :param sig_muts: list of mutations that determine the candidate child nodes
    :type sig_muts: list
    :return: candidate parent id and children id lists
    :rtype: dict
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
                # muts_no_wt = [x[1:] for x in muts]
                # muts = [x for x in muts_no_wt if x in sig_muts_no_wt]
                muts = [x for x in muts if x[1:] in sig_muts_no_wt]
                if len(muts) > 0:
                    child_seq = child_node.spike_seq
                    row = pd.DataFrame({
                        'parent_id': parent_id,
                        'child_id': child_id,
                        'parent_seq': parent_seq,
                        'child_seq': child_seq,
                        'muts': "; ".join(muts)
                    }, index=[i])
                    i = i + 1
                    post_cutoff = post_cutoff.append(row)
    if drop_by_muts:
        post_cutoff = post_cutoff.drop_duplicates(subset=['parent_seq', 'muts'], keep='first')
    else:
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
                    # corrected_mut_map[corrected_mut] = mut
                    corrected_mut_map[corrected_mut] = ref_mut
            if len(corrected_mut_map) > 0:
                corrected_muts = list(corrected_mut_map.keys())
                child_dict = {
                    'child_id': child_id,
                    'node_muts': node_muts,
                    # 'aln_muts': aln_muts,
                    'corrected_muts': corrected_muts,
                    'corrected_mut_map': corrected_mut_map
                }
                if train_seq is not None:
                    child_dict['child_in_train'] = child_seq in train_seq
                if parent_id not in parent_child:
                    parent_child[parent_id] = []
                parent_child[parent_id].append(child_dict)
    return parent_child


def freq_str(row, mut_str_col, mut_dict):
    """
    function to take a list of mutations and return their values from the mut_dict
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
    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, model_folder=None, l1_change=False,
                 cut_off='2022-1-1', load_tree=True, backend='protbert'):
        self.cut_off = cut_off
        self.backend = backend
        if 'esm' in backend:
            self.method = 'esm'
        else:
            self.method = 'protbert'
        self.tree_version = tree_version
        self.finetuned = finetuned
        self.forward_mode = forward_mode
        self.data_folder = data_folder
        self.l1_change = l1_change
        self.seq_change_path = None
        self.seq_prob_path = None
        self.seq_attn_path = None
        self.exp_folder = self.data_folder + "/exp_settings"
        if model_folder is not None:
            self.model_folder = model_folder
        else:
            file = open(self.exp_folder + "/model_folder.txt", "r")
            self.model_folder = file.read()
            file.close()
        self.tree_nodes = None
        self.model_path = None
        self.load_data(load_tree=load_tree)

    def load_data(self, load_tree=True):
        tree_folder = self.data_folder + '/treebuild'
        # load tree_nodes
        nodes_path = tree_folder + '/tree_nodes_v{}.pkl'.format(self.tree_version)
        if load_tree:
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
        self.seq_change_path = self.data_folder + '/tree_v{}_seq_{}_change_{}.pkl'.format(self.tree_version, norm,
                                                                                          train_str)
        self.seq_prob_path = self.data_folder + '/tree_v{}_seq_{}_prob_{}.pkl'.format(self.tree_version, mode_str,
                                                                                      train_str)
        self.seq_attn_path = self.data_folder + '/tree_v{}_seq_attn_{}.pkl'.format(self.tree_version, train_str)


class ParentChildMutateExp(BioTransExpSettings):

    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, model_folder=None, l1_change=False,
                 cut_off='2022-1-1', backend='protbert'):
        super().__init__(tree_version, data_folder, finetuned, forward_mode, model_folder, l1_change, cut_off, backend=backend)
        self.reference_only = False
        self.train_seq = None
        self.mutation_data = None
        self.new_mutations = None
        # self.ref_results = None
        self.ref_exp_muts = []
        self.new_muts = []
        self.old_muts = []
        # dictionary of key= parent sequence, value = list of parent IDs
        self.seq_parents = {}
        # dictionary of key = hash, value = parent sequence, can use with seqs_parents to get origional parent IDs
        self.hash_seq = {}
        # dictionary of key = sequences, hash = parent sequence, opposite of hash_seq
        self.seq_hash = {}
        self.parent_child = {}
        self.seq_probabilities = {}
        self.seq_change = {}
        self.seq_attn = {}
        self.results = pd.DataFrame()
        self.results_summary = pd.DataFrame()
        self.prep_data()
        self.seqs = []

    def prep_data(self):
        print('loading tree data')
        self.load_exp_data()
        print('prepping parent child')
        self.prep_parent_child()

    def run_experiment(self, csv=True, save_class=False, include_change=True, include_attn=False, save_name=None,
                       load_previous=False, subset_parentids=None, seq_values_only=False, seq_values_kwargs=None,
                       mutate_parents_kwargs=None):
        """
        seq_values_kwargs include: prob_seq_batchsize (chunksize for sequences in probabilities); prob_batchsize
        (batchsize for biotrans compute probabiltiies); seq_batchsize (batches for embdding); embedding_batchsize
        (batchsize for biotrans embedding)

        :param mutate_parents_kwargs: parameters for mutate_parents (ranking values)
        :type mutate_parents_kwargs: dict
        :param seq_values_kwargs: parameters to calculate sequence values
        :type seq_values_kwargs: dict
        :param include_attn:
        :type include_attn:
        :param seq_values_only:
        :type seq_values_only:
        :param save_name: save name if loading different save base name
        :type save_name: str
        :param subset_parentids: list of parentids to subset experiment to
        :type subset_parentids: list
        :param load_previous: load previous saved results, skips experiment
        :type load_previous: bool
        :param include_change: if true, calculate embedding semantic change
        :type include_change: bool
        :param csv: if true, save data in CSV format
        :type csv: bool
        :param save_class: if true, pickle the whole class
        :type save_class: bool
        :return:
        :rtype:
        """
        if subset_parentids is not None:
            self.parent_child = {k: self.parent_child[k] for k in subset_parentids}
            self.parent_seq_map()
        self.sequence_language_model_values(include_change=include_change, include_attn=include_attn,
                                            **(seq_values_kwargs or {}))
        if not seq_values_only:
            self.calc_results(csv=csv, save_class=save_class, include_change=include_change, include_attn=include_attn,
                              save_name=save_name, load_previous=load_previous,
                              mutate_parents_kwargs=mutate_parents_kwargs)

    def calc_results(self, csv=True, save_class=False, include_change=True, include_attn=False, save_name=None,
                     load_previous=False, mutate_parents_kwargs=None):
        if load_previous:
            self.load_exp_results(save_name=save_name, csv=csv)
        else:
            self.mutate_parents(include_change=include_change, include_attn=include_attn,
                                **(mutate_parents_kwargs or {}))
            self.mut_summary()
            self.save_data(csv=csv, save_class=save_class, save_name=save_name)

    def load_exp_data(self):
        # load leaf mutt dict
        tree_df_path = self.exp_folder + '/mutation_data.pkl'
        print(tree_df_path)
        self.mutation_data = pd.read_pickle(tree_df_path)
        self.new_mutations = self.mutation_data[(self.mutation_data['first_date'] >= self.cut_off)]
        self.new_mutations['pos'] = self.new_mutations['pos'].astype(int)
        self.new_muts = self.new_mutations['mutation'].values.tolist()
        self.old_muts = self.mutation_data[~(self.mutation_data.mutation.isin(self.new_muts))][
            'mutation'].values.tolist()
        self.ref_exp_muts = self.new_mutations[(self.new_mutations['mut_type'] == 'sub')]['mutation'].values.tolist()
        # load train_seq
        train_seq_path = self.exp_folder + '/train_seq.pkl'
        with open(train_seq_path, 'rb') as file:
            self.train_seq = pickle.load(file)

    def prep_parent_child(self):
        self.parent_child = candidates_by_mutations(self.tree_nodes, self.ref_exp_muts, drop_by_muts=True)
        self.parent_child = realign_candidates(self.parent_child, self.tree_nodes, self.ref_exp_muts, self.train_seq)
        # get the counter maps
        all_mut_cnt = self.mutation_data.set_index(['mutation']).n_times.to_dict()
        for parent_id, children in self.parent_child.items():
            for child_meta in children:
                mapped = child_meta['corrected_mut_map']
                # mapped_cnt is key = significant mutation, value = frequency of reference mapped mutation
                child_meta['mapped_cnt'] = {}
                for aln, aln_mapped in mapped.items():
                    cnt = all_mut_cnt[aln_mapped]
                    child_meta['mapped_cnt'][aln] = cnt
        # sub records in for testing
        # keys = list(self.parent_child.keys())[0:10]
        # self.parent_child = {key: self.parent_child[key] for key in keys}
        self.parent_seq_map()

    def parent_seq_map(self):
        self.seq_parents = {}
        for parent_id, children in self.parent_child.items():
            s = self.tree_nodes[parent_id].spike_seq
            if s not in self.seq_parents:
                self.seq_parents[s] = []
            self.seq_parents[s].append(parent_id)
        self.hash_seq = {}
        for seq in list(self.seq_parents.keys()):
            key = int(hashlib.sha1(seq.encode("utf-8")).hexdigest(), 16) % (10 ** 8)
            self.hash_seq[key] = seq
        self.seq_hash = dict((v, k) for k, v in self.hash_seq.items())

    def get_sequences(self):
        for node_id, node in self.tree_nodes.items():
            if node.root:
                self.seqs.append(node.spike_seq)
        if not self.reference_only:
            for k, v in self.parent_child.items():
                seq = self.tree_nodes[k].spike_seq
                self.seqs.append(seq)
            self.seqs = list(set(self.seqs))
            print('Total Unique Parent Sequences: {}'.format(len(self.seqs)))

    def sequence_language_model_values(self, list_seq_batchsize=15, prob_batchsize=20,
                                       seq_batchsize=400, embedding_batchsize=40, include_change=True,
                                       run_chunked_change=False, combine=False, include_attn=False):
        self.get_sequences()
        if include_attn:
            print('checking saved attention changes')
            seqs_for_attn, self.seq_attn = find_previous_saved(self.seqs, self.seq_attn_path)
            if len(seqs_for_attn) > 0:
                self.parent_attn_change_batched(combine=combine)

        print('checking saved probabilities')
        seqs_for_proba, self.seq_probabilities = find_previous_saved(self.seqs, self.seq_prob_path)
        if include_change:
            print('checking saved embedding change')
            seqs_for_change, self.seq_change = find_previous_saved(self.seqs, self.seq_change_path)
        else:
            seqs_for_change = []
        if len(seqs_for_proba) > 0 or len(seqs_for_change) > 0:
            bio_trans = load_biotrans(model_path=self.model_path, backend=self.backend)
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

    def parent_attn_change_batched(self, pool_heads_max=True, pool_layer_max=True, l1_norm=False, combine=False):
        str_id = uuid.uuid1()
        attn_folder = self.data_folder + '/attn_changes'
        check_directory(attn_folder)
        suffix = self.seq_attn_path.split('/')[-1]
        attn_save_files = folder_file_list(attn_folder, suffix)
        current_save_path = attn_folder + '/{}_{}'.format(str_id, suffix)
        seq_dict = combine_seq_dicts(file_list=attn_save_files)
        self.seq_attn.update(seq_dict)
        seqs_for_attn = []
        for seq in self.seqs:
            if seq not in self.seq_attn:
                seqs_for_attn.append(seq)
        if len(seqs_for_attn) > 0:
            print('computing attention change for {} sequences'.format(len(seqs_for_attn)))
            current_seq_attn = {}
            current_seq_attn = attention_change_batchs(seqs=seqs_for_attn, model_path=self.model_path,
                                                       seq_attn=current_seq_attn,
                                                       save_path=current_save_path,
                                                       pool_heads_max=pool_heads_max,
                                                       pool_layer_max=pool_layer_max,
                                                       l1_norm=l1_norm,
                                                       method=self.method,
                                                       backend=self.backend)
            print('Saving {} sequence attention change'.format(len(current_seq_attn)))
            with open(current_save_path, 'wb') as a:
                pickle.dump(current_seq_attn, a)
            self.seq_attn.update(current_seq_attn)
        if combine:
            if os.path.isfile(self.seq_attn_path):
                attn_save_files.append(self.seq_attn_path)
            seq_dict = combine_seq_dicts(file_list=attn_save_files)
            self.seq_attn.update(seq_dict)
            print('Saving {} combined sequence attention change'.format(len(self.seq_attn)))
            with open(self.seq_attn_path, 'wb') as a:
                pickle.dump(self.seq_attn, a)

    def parent_semantic_change_batched(self, bio_trans, seq_batchsize, embedding_batchsize, list_seq_batchsize,
                                       combine=False):
        str_id = uuid.uuid1()
        if self.l1_change:
            change_folder = self.data_folder + "/l1_changes"
        else:
            change_folder = self.data_folder + '/l2_changes'
        check_directory(change_folder)
        suffix = self.seq_change_path.split('/')[-1]
        change_save_files = folder_file_list(change_folder, suffix)
        current_save_path = change_folder + '/{}_{}'.format(str_id, suffix)

        seq_dict = combine_seq_dicts(file_list=change_save_files)
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
            seq_dict = combine_seq_dicts(file_list=change_save_files)
            self.seq_change.update(seq_dict)
            print('Saving {} combined sequence semantic change'.format(len(self.seq_change)))
            with open(self.seq_change_path, 'wb') as a:
                pickle.dump(self.seq_change, a)

    def prep_seq_mutation_dict(self, seq, sig_muts, include_change=True, include_attn=True):
        probabilities = self.seq_probabilities[seq]
        if include_change:
            changes = self.seq_change[seq]
        else:
            changes = None
        if include_attn:
            attn = self.seq_attn[seq]
        else:
            attn = None
        seq_mutations = get_seq_mutation_dict(seq=seq, probabilities=probabilities, changes=changes,
                                              significant_mutations=sig_muts, attn=attn)
        return seq_mutations

    def prep_parent_seq(self, parent_seq, include_change=True, include_attn=True):
        parent_ids = self.seq_parents[parent_seq]
        sig_muts = []
        mut_map = {}
        mapped_cnt = {}
        for parent_id in parent_ids:
            children = self.parent_child[parent_id]
            for child in children:
                if child['child_in_train']:
                    continue
                sig_muts.extend(child['corrected_muts'])
                mut_map.update(child['corrected_mut_map'])
                mapped_cnt.update(child['mapped_cnt'])
        sig_muts = [x for x in sig_muts if x in mut_map and mut_map[x] in self.ref_exp_muts]
        sig_muts = list(set(sig_muts))
        seq_mutations = self.prep_seq_mutation_dict(parent_seq, sig_muts, include_change=include_change,
                                                    include_attn=include_attn)
        return seq_mutations, sig_muts, mut_map, mapped_cnt

    def mutate_single_parent(self, parent_seq, include_change=True, include_attn=True, **ranking_values):
        data_list = []
        seq_mutations, sig_muts, mut_map, mapped_cnt = self.prep_parent_seq(parent_seq=parent_seq,
                                                                            include_change=include_change,
                                                                            include_attn=include_attn)
        if len(sig_muts) < 1:
            return data_list
        # run with all mutations
        parent_hash = self.seq_hash[parent_seq]
        parent_in_train = parent_seq in self.train_seq
        data_list = results_over_thresholds(seq_mutations, sig_muts, parent_hash,
                                            mapped_cnt=mapped_cnt, mut_map=mut_map,
                                            parent_in_train=parent_in_train, **ranking_values)

        return data_list

    def mutate_parents(self, include_change=True, include_attn=True, **ranking_values):
        data_list = []
        for parent_seq, parent_ids in tqdm(self.seq_parents.items(), desc='Unique Parent Sequences'):
            parent_data_list = self.mutate_single_parent(parent_seq, include_change=include_change,
                                                         include_attn=include_attn, **ranking_values)
            data_list.extend(parent_data_list)
        self.results = pd.DataFrame(data_list)

    def mut_summary(self):
        df = self.results[(self.results['result_type'] == 'Solo Mutation')]
        agg_dict = {'cscs_auc': ['mean', 'std'], 'dnms_auc': ['mean', 'std'],
                    'change_auc': ['mean', 'std'], 'prob_auc': ['mean', 'std'],
                    'attn_auc': ['mean', 'std'],
                    'ref_muts': 'nunique', 'parent_hash': 'count'}
        agg_dict = {k: agg_dict[k] for k in agg_dict if k in df}
        measures = ['cscs_auc', 'change_auc', 'prob_auc', 'dnms_auc', 'attn_auc']
        measures = [x for x in measures if x in df]
        thresholds = sorted(list(set(df['threshold'])))
        self.results_summary = pd.DataFrame()
        for threshold in thresholds:
            df1 = pd.DataFrame(df[(df['threshold'] >= threshold)].agg(agg_dict))
            df1[['ref_muts', 'parent_hash']] = df1[['ref_muts', 'parent_hash']].bfill()
            df1 = df1.T
            df2 = pd.DataFrame(df1['mean']).T.reset_index(drop=True)
            df3 = pd.DataFrame(df1[(df1.index.isin(measures))]['std']).T.reset_index(drop=True)
            df3.columns = [x + "_std" for x in df3.columns]
            df2 = df2.join(df3)
            df2.insert(0, 'threshold', threshold)
            self.results_summary = self.results_summary.append(df2)
        self.results_summary.rename(columns={'ref_muts': 'n_muts', 'parent_hash': 'n_parent_seq'}, inplace=True)
        # print(self.results_summary[['threshold']+measures])
        print("Average per Mutation")
        avg = self.results.groupby('result_type')[measures].mean()
        print(avg)
        print()
        print("Average per threshold")
        avg1 = self.results_summary[measures].mean()
        print(avg1)

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
        if self.reference_only:
            base_pref = 'ref'
        else:
            base_pref = 'pc'
        if save_name is not None:
            base_name = save_name
        else:
            base_name = '{}_tree_v{}_{}_{}_{}_mode'.format(base_pref, self.tree_version, train_str, norm, mode_str)
        return base_name

    def save_data(self, csv=True, save_class=False, save_name=None):
        save_folder = self.data_folder + "/results"
        base_name = self.get_savename(save_name=save_name)
        check_directory(save_folder)
        if csv:
            file_name = base_name + '.csv'
            summary_file_name = base_name + '_summary.csv'
            self.results.to_csv(save_folder + '/' + file_name, index=False)
            self.results_summary.to_csv(save_folder + '/' + summary_file_name, index=False)
        else:
            file_name = base_name + '.pkl'
            summary_file_name = base_name + "_summary.pkl"
            self.results.to_pickle(save_folder + '/' + file_name)
            self.results_summary.to_pickle(save_folder + '/' + summary_file_name)
        if save_class:
            with open(base_name + '_MutateParentChild.pkl', 'wb') as file:
                pickle.dump(self, file)

    def load_exp_results(self, save_name=None, csv=False):
        save_folder = self.data_folder + "/results"
        base_name = self.get_savename(save_name=save_name)
        if csv:
            file_name = base_name + '.csv'
            summary_file_name = base_name + '_summary.csv'
            try:
                self.results = pd.read_csv(save_folder + '/' + file_name)
                self.results_summary = pd.read_csv(save_folder + '/' + summary_file_name)
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
    parser.add_argument('--include_attn', action='store_true',
                        help='include attn in experiment')
    parser.add_argument('--seq_values_only', action='store_true',
                        help='only calculate sequence values, skips getting results')
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
    parser.add_argument('--attn_seq_batchsize', type=int, default=2,
                        help='sequence list batch for Attention')
    parser.add_argument('--embedding_batchsize', type=int, default=50,
                        help='mini sequence batchsize for embedding')
    parser.add_argument('--sub_parentid_path', type=str, default=None,
                        help='if provided, load a list of parent_ids to subset the experiment to')
    parser.add_argument('--csv', action='store_true',
                        help='if provided, save results in CSV format.')
    parser.add_argument('--load_previous', action='store_true',
                        help='if provided, save results in CSV format.')
    parser.add_argument('--alpha', type=float, default=1.5,
                        help='Semantic Change weighting value for DNMS.')
    parser.add_argument('--beta', type=float, default=3.0,
                        help='Grammaticality weighting value for DNMS.')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Attention Change weighting value for DNMS.')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    parent_child_exp = ParentChildMutateExp(tree_version=args.tree_version,
                                            finetuned=args.finetuned,
                                            forward_mode=args.masked_mode,
                                            data_folder=args.data_folder,
                                            l1_change=args.l2_norm)

    print("Finetuned is: {}".format(parent_child_exp.finetuned))
    print("Forward Mode is: {}".format(parent_child_exp.forward_mode))
    print("L1 change is: {}".format(parent_child_exp.l1_change))

    if args.sub_parentid_path is not None:
        with open(args.sub_parentid_path, 'rb') as f:
            sub_parentids = pickle.load(f)
    else:
        sub_parentids = None

    seq_value_params = {'list_seq_batchsize': args.list_seq_batchsize,
                        'prob_batchsize': args.prob_batchsize,
                        'seq_batchsize': args.seq_batchsize,
                        'embedding_batchsize': args.embedding_batchsize,
                        'run_chunked_change': args.change_batched,
                        'combine': args.combine}

    print(seq_value_params)

    mutate_params = {'alpha': args.alpha,
                     'beta': args.beta,
                     'gamma': args.gamma}

    print(mutate_params)

    parent_child_exp.run_experiment(include_change=args.include_change, include_attn=args.include_attn,
                                    seq_values_only=args.seq_values_only, csv=args.csv,
                                    subset_parentids=sub_parentids, seq_values_kwargs=seq_value_params,
                                    load_previous=args.load_previous, mutate_parents_kwargs=mutate_params)
