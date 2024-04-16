import os.path
from BioTransLanguageModel import *
from DataHelpers import read_tsv_file
from NCBIData import spike_muts_only
from ParentChildMutate import BioTransExpSettings
from MutationHelpers import *
import argparse
from scipy.stats import spearmanr


def get_rbd_pos_dict_mutations(df):
    mutations = df['mutant'].values.tolist()
    df['site'] = df['site'] - 1
    df = df[['wildtype', 'site']].drop_duplicates()
    rbd_pos = pd.Series(df['wildtype'].values, index=df['site']).to_dict()
    return rbd_pos, mutations


def chk_rbd_pos_seq(seq, rbd_pos):
    valid = True
    for pos, aa in rbd_pos.items():
        if seq[pos] != aa:
            valid = False
    return valid


def get_corrected_mutations(ref_seq, mutations_from_ref, mutations_to_correct):
    map_ref_pos = ref_seq_pos_map(ref_seq, mutations=mutations_from_ref)
    corrected_mutations = []
    for mut in mutations_to_correct:
        wt, pos, mutted = pullout_pos_mut(mut)
        seq_pos = map_ref_pos[pos]
        corrected_mut = "{}{}{}".format(wt, seq_pos + 1, mutted)
        corrected_mutations.append(corrected_mut)
    return corrected_mutations


def get_corr_df(rbd_data, strain):
    mutation_score = rbd_data[(rbd_data['target_name'] == strain)]
    cols = ['binding', 'expression']
    rows = ['prob', 'change', 'attn']
    corr_df = pd.DataFrame(index=rows, columns=cols, dtype=float)
    pval_df = pd.DataFrame(index=rows, columns=cols, dtype=float)

    n_tests = 0
    for i, col in enumerate(cols):
        for j, row in enumerate(rows):
            n_tests += 1
            data = mutation_score[(mutation_score[col].notnull()) & (mutation_score[row].notnull())]
            corr_coef, p_value = spearmanr(data[col], data[row])
            corr_df.iloc[j, i] = corr_coef
            pval_df.iloc[j, i] = p_value
    corrected_pval = pval_df / n_tests
    sig_threshold = 0.05
    # sig_mask creates NaN where values are not significant
    sig_mask = corrected_pval < sig_threshold
    sig_corr_df = corrected_pval.copy()
    sig_corr_df[~sig_mask] = np.nan
    corr = prep_corr_df(corr_df)
    sig = prep_corr_df(sig_corr_df)
    result = pd.merge(corr, sig, on=['Fitness Measure', 'variable'], suffixes=('', '_pval'))
    result.insert(0, 'target_name', strain)
    return result


def prep_corr_df(corr_df):
    df = corr_df.T.reset_index()
    df.columns = ['Fitness Measure', 'Grammaticality', 'Semantic Change',
                  'Attention Change']
    df1 = pd.melt(df, id_vars=['Fitness Measure'])
    df1['Fitness Measure'] = df1['Fitness Measure'].str.capitalize()
    return df1


class BioTransFitness(BioTransExpSettings):
    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, l1_change=False):
        super().__init__(tree_version, data_folder, finetuned=finetuned, forward_mode=forward_mode,
                         l1_change=l1_change, load_tree=False)
        self.rbd_data_folder = self.data_folder + '/rbd_exp_omicron'
        self.rbd_data = None
        self.exp_data = None
        self.target_seq_meta = {}
        self.seqs = []
        self.seq_probabilities = {}
        self.seq_change = {}
        self.seq_attn = {}
        self.record_dict = {}
        self.seq_muts_from_ref = {}
        self.clade_record_dict = {'WT': 'Wuhan/Hu-1/2019',
                                  'BA.1': 'hCoV-19/England/LSPA-363D038/2022|EPI_ISL_10000028|2022-02-07',
                                  'BA.2': 'hCoV-19/Scotland/QEUH-363F151/2022|EPI_ISL_10000005|2022-02-08'
                                  }
        self.strains_files = {'WT': 'bind_expr_WT.csv',
                              'BA.1': 'bind_expr_BA1.csv',
                              'BA.2': 'bind_expr_BA2.csv',
                              }
        self.adjusted_muts = {}

    def run_experiment(self, include_calcs=None, list_seq_batchsize=15, prob_batchsize=20, seq_batchsize=400,
                       embedding_batchsize=40, pool_heads_max=True, pool_layer_max=True, l1_norm=False):
        self.prep_data()
        self.language_model_values(include_calcs=include_calcs, list_seq_batchsize=list_seq_batchsize,
                                   prob_batchsize=prob_batchsize, seq_batchsize=seq_batchsize,
                                   embedding_batchsize=embedding_batchsize, pool_heads_max=pool_heads_max,
                                   pool_layer_max=pool_layer_max, l1_norm=l1_norm)
        self.rbd_data_results()

    def prep_exp_data(self):
        self.exp_data = pd.DataFrame()
        for strain, file in self.strains_files.items():
            file_path = self.rbd_data_folder + '/exp_data/' + file
            data = pd.read_csv(file_path)
            data = data[~(data['mutation'] == '*')]
            if strain in self.adjusted_muts:
                data = data.assign(bias_e=0.0, bias_b=0.0)
                for site, mut in self.adjusted_muts[strain]:
                    expr = data.query('site == @site and mutation == @mut')['expr_avg'].item()
                    bind = data.query('site == @site and mutation == @mut')['bind_avg'].item()
                    data.loc[data['site'] == site, 'bias_e'] += expr
                    data.loc[data['site'] == site, 'bias_b'] += bind
                    data.loc[data['site'] == site, 'wildtype'] = mut
                data['expr_avg'] -= data['bias_e']
                data['bind_avg'] -= data['bias_b']
                data = data.drop(columns=['bias_e', 'bias_b'])
                data['mutant'] = data['wildtype'] + data['site'].astype(str) + data['mutation']
            data = data[(data['wildtype'] != data['mutation'])]
            data.insert(0, 'target_name', strain)
            self.exp_data = self.exp_data.append(data)

    def build_target_seq_meta(self):
        ref_seq = load_ref_spike()
        ref_seq = str(ref_seq.seq)
        self.target_seq_meta = {}
        strain = 'WT'
        data = self.exp_data[(self.exp_data['target_name'] == strain)]
        rbd_pos, tested_mutations = get_rbd_pos_dict_mutations(data)
        if not chk_rbd_pos_seq(ref_seq, rbd_pos):
            raise ValueError("Sequence doesn't match RBD WT from Experiment, check {}".format(strain))
        seq_meta = {'strain': strain, 'seq_id': self.clade_record_dict[strain], 'tested_muts': tested_mutations}
        self.target_seq_meta[ref_seq] = seq_meta
        omicron_strains = ['BA.1', 'BA.2']
        for strain in omicron_strains:
            data = self.exp_data[(self.exp_data['target_name'] == strain)]
            rbd_pos, tested_mutations = get_rbd_pos_dict_mutations(data)
            record_id = self.clade_record_dict[strain]
            seq = str(self.record_dict[record_id].seq)
            seq = remove_end_star(seq)
            if not chk_rbd_pos_seq(seq, rbd_pos):
                raise ValueError("Sequence doesn't match RBD WT from Experiment, check {}".format(strain))
            muts_from_ref = self.seq_muts_from_ref[record_id]
            seq = seq.replace('-', '')
            if not check_known_aa(seq):
                raise ValueError("Sequence has unknown AA, check {}".format(strain))
            corrected_mutations = get_corrected_mutations(ref_seq, muts_from_ref, tested_mutations)
            seq_meta = {'strain': strain,
                        'seq_id': record_id,
                        'tested_muts': corrected_mutations,
                        'orig_tested_muts': tested_mutations}
            self.target_seq_meta[seq] = seq_meta

        target_seq_meta_path = self.rbd_data_folder + "/seq_meta.pkl"
        with open(target_seq_meta_path, 'wb') as f:
            pickle.dump(self.target_seq_meta, f)

    def load_seq_data(self):
        fasta_path = self.rbd_data_folder + "/nextclade_omicron_gene_S.fasta"
        self.record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))
        df = read_tsv_file(self.rbd_data_folder + '/nextclade_omicron.tsv')
        aa_col = ['aaSubstitutions', 'aaDeletions', 'aaInsertions']
        for col in aa_col:
            df[col + '_spike'] = df[col].apply(lambda x: spike_muts_only(x))
        df['spike_mutations'] = df['aaSubstitutions_spike'] + df['aaDeletions_spike']
        self.seq_muts_from_ref = pd.Series(df['spike_mutations'].values, index=df['seqName']).to_dict()

    def prep_data(self):
        self.seq_prob_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_prob_path.split('seq')[-1])
        self.seq_change_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_change_path.split('seq')[-1])
        self.seq_attn_path = self.rbd_data_folder + "/rbd_exp{}".format(self.seq_attn_path.split('seq')[-1])
        target_seq_meta_path = self.rbd_data_folder + "/seq_meta.pkl"
        self.prep_exp_data()
        if os.path.isfile(target_seq_meta_path):
            with open(target_seq_meta_path, 'rb') as f:
                self.target_seq_meta = pickle.load(f)
        else:
            self.load_seq_data()
            self.build_target_seq_meta()
        self.seqs = list(self.target_seq_meta.keys())

    def get_subset_mutations(self, seqs):
        subset_mutations = []
        for seq in seqs:
            subset_muts = self.target_seq_meta[seq]['tested_muts']
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

    def rbd_data_results(self):
        self.rbd_data = pd.DataFrame()
        for seq, meta in self.target_seq_meta.items():
            strain = meta['strain']
            tested_muts = meta['tested_muts']
            if strain == 'WT':
                orig_tested_muts = tested_muts
            else:
                orig_tested_muts = meta['orig_tested_muts']
            mut_map = dict(zip(tested_muts, orig_tested_muts))
            file_path = self.rbd_data_folder + '/exp_data/' + self.strains_files[strain]
            data = pd.read_csv(file_path)
            data = data[(data['wildtype'] != data['mutation'])]
            data = data[~(data['mutation'] == '*')]
            binding = data[(data['bind_avg'].notnull())]
            binding = pd.Series(binding['bind_avg'].values, index=binding['mutant']).to_dict()
            binding = {k: binding[mut_map[k]] for k in mut_map if mut_map[k] in binding}
            expression = data[(data['expr_avg'].notnull())]
            expression = pd.Series(expression['expr_avg'].values, index=expression['mutant']).to_dict()
            expression = {k: expression[mut_map[k]] for k in mut_map if mut_map[k] in expression}
            attn = self.seq_attn[seq]
            change = self.seq_change[seq]
            prob = self.seq_probabilities[seq]
            seq_mutations = get_seq_mutation_dict(seq=seq, probabilities=prob, changes=change,
                                                  significant_mutations=None, attn=attn,
                                                  subset_mutations=tested_muts)
            df = pd.DataFrame(seq_mutations.values())
            df['binding'] = df['mutation'].map(binding)
            df['expression'] = df['mutation'].map(expression)
            df.insert(0, 'target_name', strain)
            self.rbd_data = self.rbd_data.append(df)
        self.rbd_data.to_pickle(self.rbd_data_folder + '/rbd_data_results.pkl')


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
