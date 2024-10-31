from ParentChildMutate import *


class RefSeqMutateExp(ParentChildMutateExp):
    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=True, model_folder=None, l1_change=True,
                 cut_off='2022-1-1', backend='protbert'):
        super().__init__(tree_version, data_folder, finetuned, forward_mode, model_folder, l1_change, cut_off, backend)
        self.reference_only = True

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
        :param include_attn: If true, include attention in experiment. Can be left out if just calculating values
        :type include_attn: bool
        :param seq_values_only: If true, only calculate sequence values, skips saving/calculating results
        :type seq_values_only: bool
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
        """
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
            self.mutate_ref(include_change=include_change, include_attn=include_attn,
                            **(mutate_parents_kwargs or {}))
            self.mut_summary()
            self.save_data(csv=csv, save_class=save_class, save_name=save_name)

    def mutate_ref(self, include_change, include_attn, **ranking_values):
        mapped_cnt = self.mutation_data.set_index(['mutation']).n_times.to_dict()
        ref_sig_muts = []
        for parent_node, children in self.parent_child.items():
            for child in children:
                ref_muts = list(child['corrected_mut_map'].values())
                ref_sig_muts.extend(ref_muts)
        sig_muts = list(set(ref_sig_muts))
        mapped_cnt = {k: mapped_cnt[k] for k in sig_muts}
        ref_seq = load_ref_spike()
        ref_seq = str(ref_seq.seq)
        seq_mutations = self.prep_seq_mutation_dict(ref_seq, sig_muts, include_change=include_change,
                                                    include_attn=include_attn)
        print('Calculating Results over Thresholds')
        results = results_over_thresholds(seq_mutations, sig_muts, 'ref_seq', mapped_cnt, mut_map=None,
                                          parent_in_train=None, **ranking_values)
        self.results = pd.DataFrame(results)


def parse_args():
    parser = argparse.ArgumentParser(description='Reference Mutate experiment')
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
    parser.add_argument('--backend', type=str,
                        help='model to load for biotransformer', default='protbert')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    ref_exp = RefSeqMutateExp(tree_version=args.tree_version,
                              finetuned=args.finetuned,
                              forward_mode=args.masked_mode,
                              data_folder=args.data_folder,
                              l1_change=args.l2_norm,
                              backend=args.backend)

    print("Finetuned is: {}".format(ref_exp.finetuned))
    print("Forward Mode is: {}".format(ref_exp.forward_mode))
    print("L1 change is: {}".format(ref_exp.l1_change))

    seq_value_params = {'list_seq_batchsize': args.list_seq_batchsize,
                        'prob_batchsize': args.prob_batchsize,
                        'seq_batchsize': args.seq_batchsize,
                        'embedding_batchsize': args.embedding_batchsize}

    print(seq_value_params)

    mutate_params = {'alpha': args.alpha,
                     'beta': args.beta,
                     'gamma': args.gamma}

    print(mutate_params)

    ref_exp.run_experiment(include_change=args.include_change, include_attn=args.include_attn,
                                    seq_values_only=args.seq_values_only, csv=args.csv,
                                    seq_values_kwargs=seq_value_params,
                                    load_previous=args.load_previous, mutate_parents_kwargs=mutate_params)
