import os.path

from BioTransLanguageModel import *
from ParentChildMutate import BioTransExpSettings
from MutationHelpers import *
import argparse


def get_rbd_pos_dict_mutations(file_path):
    df = pd.read_csv(file_path)
    df = df[(df['wildtype']!=df['mutation'])]
    df = df[~(df['mutation']=='*')]
    mutations = df['mutant'].values.tolist()
    df['site'] = df['site'] - 1
    df = df[['wildtype','site']].drop_duplicates()
    rbd_pos = pd.Series(df['wildtype'].values, index=df['site']).to_dict()
    return rbd_pos, mutations


class BioTransFitness(BioTransExpSettings):
    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, l1_change=False):
        super().__init__(tree_version, data_folder, finetuned=finetuned, forward_mode=forward_mode,
                         l1_change=l1_change, load_tree=False)
        self.rbd_data_folder = self.data_folder + '/rbd_exp_omicron'
        self.rbd_data = None
        self.target_seq_meta = {}
        self.seqs = []
        self.seq_probabilities = {}
        self.seq_change = {}
        self.seq_attn = {}

    def run_experiment(self, include_calcs=None, list_seq_batchsize=15, prob_batchsize=20, seq_batchsize=400,
                       embedding_batchsize=40, pool_heads_max=True, pool_layer_max=True, l1_norm=False):
        self.prep_data()
        self.language_model_values(include_calcs=include_calcs, list_seq_batchsize=list_seq_batchsize,
                                   prob_batchsize=prob_batchsize, seq_batchsize=seq_batchsize,
                                   embedding_batchsize=embedding_batchsize, pool_heads_max=pool_heads_max,
                                   pool_layer_max=pool_layer_max, l1_norm=l1_norm)

    def build_target_seq_meta(self):
        self.load_data(load_tree=True)
        strains_files = {'WT': 'bind_expr_WT.csv', 'BA.1': 'bind_expr_BA1.csv', 'BA.2': 'bind_expr_BA2.csv'}
        ref_seq = load_ref_spike()
        ref_seq = str(ref_seq.seq)
        strain = 'WT'
        path = strains_files[strain]
        file_path = self.rbd_data_folder + '/exp_data/' + path
        _, mutations = get_rbd_pos_dict_mutations(file_path)
        seq_meta = {
            'tested_muts': mutations,
            'node_ids': ['Wuhan/Hu-1/2019']
        }
        self.target_seq_meta[strain] = {ref_seq: seq_meta}
        omicron_strains = ['BA.1', 'BA.2']
        for strain in omicron_strains:
            path = strains_files[strain]
            self.target_seq_meta[strain] = {}
            file_path = self.rbd_data_folder + '/exp_data/' + path
            rbd_pos, tested_mutations = get_rbd_pos_dict_mutations(file_path)
            seq_ids = {}
            for node_id, node in self.tree_nodes.items():
                if 'pango_lineage_local' in node.node_attrs and \
                        node.node_attrs['pango_lineage_local'][ 'value'] == strain:
                    muts = node.spike_mutations
                    seq = node.spike_seq
                    del_seq = mutate_sequence(ref_seq, muts, ignore_deletion=False)
                    valid = True
                    for pos, aa in rbd_pos.items():
                        if del_seq[pos] != aa:
                            valid = False
                    if valid:
                        if seq not in seq_ids:
                            seq_ids[seq] = []
                        seq_ids[seq].append(node_id)
            for seq, node_ids in seq_ids.items():
                seq_meta = {}
                seq_meta['orig_tested_muts'] = tested_mutations
                seq_meta['node_ids'] = node_ids
                node = self.tree_nodes[node_ids[0]]
                map_ref_pos = ref_seq_pos_map(ref_seq, mutations=node.spike_mutations)
                corrected_mutations = []
                for mut in tested_mutations:
                    wt, pos, mutted = pullout_pos_mut(mut)
                    seq_pos = map_ref_pos[pos]
                    corrected_mut = "{}{}{}".format(wt, seq_pos + 1, mutted)
                    corrected_mutations.append(corrected_mut)
                seq_meta['tested_muts'] = corrected_mutations
                self.target_seq_meta[strain][seq] = seq_meta
        target_seq_meta_path = self.rbd_data_folder + "/seq_meta.pkl"
        with open(target_seq_meta_path, 'wb') as f:
            pickle.dump(self.target_seq_meta, f)


    def prep_data(self):
        self.seq_prob_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_prob_path.split('seq')[-1])
        self.seq_change_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_change_path.split('seq')[-1])
        self.seq_attn_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_attn_path.split('seq')[-1])
        target_seq_meta_path = self.rbd_data_folder+"/seq_meta.pkl"
        if os.path.isfile(target_seq_meta_path):
            with open(target_seq_meta_path, 'rb') as f:
                self.target_seq_meta = pickle.load(f)
        else:
            self.build_target_seq_meta()
        self.seqs = []
        for strain, seq_meta in self.target_seq_meta.items():
            self.seqs.extend(list(seq_meta.keys()))
        self.seqs = list(set(self.seqs))

    def get_subset_mutations(self, seqs):
        full_seq_meta = {}
        for strain, seq_meta in self.target_seq_meta.items():
            full_seq_meta.update(seq_meta)
        subset_mutations = []
        for seq in seqs:
            subset_muts = full_seq_meta[seq]['tested_muts']
            subset_mutations.append(subset_muts)
        return subset_mutations

    def language_model_values(self, list_seq_batchsize=15, prob_batchsize=20, seq_batchsize=400,
                              embedding_batchsize=40, include_calcs=None, pool_heads_max=True,
                              pool_layer_max=True, l1_norm=False):
        if include_calcs is None:
            include_calcs = ['attn', 'prob', 'change']
        if 'attn' in include_calcs:
            self.calc_seq_attn(pool_heads_max=pool_heads_max, pool_layer_max=pool_layer_max,
                               l1_norm=l1_norm)

        # best to do attention first with multi-processing and then probabilities and semantic change with single GPU
        if 'prob' in include_calcs:
            print('checking saved probabilities')
            seqs_for_proba, self.seq_probabilities = find_previous_saved(self.seqs, self.seq_prob_path)
        else:
            seqs_for_proba = []
        if 'change' in include_calcs:
            print('checking saved embedding change')
            seqs_for_change, self.seq_change = find_previous_saved(self.seqs, self.seq_change_path)
        else:
            seqs_for_change = []
        if len(seqs_for_proba) > 0 or len(seqs_for_change) > 0:
            bio_trans = load_biotrans(model_path=self.model_path)
            if len(seqs_for_proba) > 0:
                print('computing probabilities for {} sequences'.format(len(seqs_for_proba)))
                # if only doing forward mode, it's not that important to subset the mutations.
                self.seq_probabilities = compute_probabilities(bio_trans=bio_trans, seqs=seqs_for_proba,
                                                               batchsize=prob_batchsize, chunksize=list_seq_batchsize,
                                                               forward_mode=self.forward_mode,
                                                               prior_seq_probabilities=self.seq_probabilities,
                                                               save_path=self.seq_prob_path)
                print('Saving {} sequence probabilities'.format(len(self.seq_probabilities)))
                with open(self.seq_prob_path, 'wb') as a:
                    pickle.dump(self.seq_probabilities, a)
            if len(seqs_for_change) > 0:
                print('computing semantic change for {} sequences'.format(len(seqs_for_change)))
                subset_mutations = self.get_subset_mutations(seqs_for_change)
                seq_subset_mut_dict = dict(zip(seqs_for_change, subset_mutations))
                self.seq_change = embedding_change_batchs(seqs=seqs_for_change, bio_trans=bio_trans,
                                                          seq_batchsize=seq_batchsize,
                                                          embedding_batchsize=embedding_batchsize,
                                                          seq_change=self.seq_change,
                                                          save_path=self.seq_change_path,
                                                          chunksize=list_seq_batchsize,
                                                          l1_norm=self.l1_change,
                                                          seq_subset_mut_dict=seq_subset_mut_dict)
                print('Saving {} sequence semantic change'.format(len(self.seq_change)))
                with open(self.seq_change_path, 'wb') as a:
                    pickle.dump(self.seq_change, a)

    def calc_seq_attn(self, pool_heads_max=True, pool_layer_max=True, l1_norm=False):
        print('checking saved attention changes')
        seqs_for_attn, self.seq_attn = find_previous_saved(self.seqs, self.seq_attn_path)
        subset_mutations = self.get_subset_mutations(seqs_for_attn)
        if len(seqs_for_attn) > 0:
            self.seq_attn = attention_change_batchs(seqs=seqs_for_attn, model_path=self.model_path,
                                                    seq_attn=self.seq_attn,
                                                    save_path=self.seq_attn_path,
                                                    pool_heads_max=pool_heads_max,
                                                    pool_layer_max=pool_layer_max, l1_norm=l1_norm,
                                                    subset_mutations=subset_mutations)
            print('Saving {} sequence attention change'.format(len(self.seq_attn)))
            with open(self.seq_attn_path, 'wb') as a:
                pickle.dump(self.seq_attn, a)
            print('Done with Calculating Attention')


def parse_args():
    parser = argparse.ArgumentParser(description='RBD Fitness analysis experiment')
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
    parser.add_argument('--include_prob', action='store_true',
                        help='include prob in experiment')
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
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    args = parse_args()
    print(args)
    
    fitness_exp = BioTransFitness(tree_version=args.tree_version,
                                  finetuned=args.finetuned,
                                  forward_mode=args.masked_mode,
                                  data_folder=args.data_folder,
                                  l1_change=args.l2_norm)

    print("masked mode arg: {}".format(args.masked_mode))
    print("Finetuned is: {}".format(fitness_exp.finetuned))
    print("Forward Mode is: {}".format(fitness_exp.forward_mode))
    print("L1 change is: {}".format(fitness_exp.l1_change))

    calcs = []
    if args.include_attn:
        calcs.append('attn')
    if args.include_prob:
        calcs.append('prob')
    if args.include_change:
        calcs.append('change')

    params = {'list_seq_batchsize': args.list_seq_batchsize,
              'prob_batchsize': args.prob_batchsize,
              'seq_batchsize': args.seq_batchsize,
              'embedding_batchsize': args.embedding_batchsize}

    print(params)
    print(calcs)
    fitness_exp.run_experiment(include_calcs=calcs, **params)




