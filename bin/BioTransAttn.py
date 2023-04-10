from ParentChildMutate import *
from scipy import special
from BioTransLanguageModel import *


def parse_args():
    parser = argparse.ArgumentParser(description='Biotransformer Attention Experiment')
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
    parser.add_argument('--save_folder', type=str,
                        help='Folder for saving data')
    parser.add_argument('--seq_path', type=str, default=None,
                        help='if provided, load a sequence in string format, '
                             'otherwise it will use reference')
    arguments = parser.parse_args()
    return arguments


def normalize_attn_mat(attn):
    attn_sum = attn.sum()
    norm_attn = attn/attn_sum
    return norm_attn


def kl_divergence(mat1, mat2):
    k1 = special.kl_div(mat1, mat2)
    k2 = special.kl_div(mat2, mat1)
    div = ((k1 + k2)/2)
    summed_div = div.sum()
    return summed_div


if __name__ == '__main__':
    args = parse_args()

    print(args)

    pc = MutateParentChild(tree_version=args.tree_version,
                           finetuned=args.finetuned,
                           forward_mode=args.masked_mode,
                           data_folder=args.data_folder,
                           l1_change=args.l2_norm)
    print("masked mode arg: {}".format(args.masked_mode))
    print("Finetuned is: {}".format(pc.finetuned))
    print("Forward Mode is: {}".format(pc.forward_mode))
    print("L1 change is: {}".format(pc.l1_change))

    pc.run_experiment(include_change=True, load_previous=True, excel=False)

    if args.seq_path is not None:
        with open(args.seq_path, 'rb') as f:
            seq = pickle.load(f)
        using_ref = False
    else:
        seq = load_ref_spike()
        seq = str(seq.seq)
        using_ref = True

    exp_folder = args.data_folder + "/exp_settings"
    file = open(exp_folder + "/model_folder.txt", "r")
    model_folder = file.read()
    file.close()

    model_path = last_biotrans_ft_path(model_folder)

    if using_ref:
        save_path = args.save_folder+"/ref_attn_dict_l2.pkl"
    else:
        save_path = args.save_folder+"/parent_seq_attn_dict_l2.pkl"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, bio_trans_model_ft = load_biotrans_for_attn(device=device, model_path=model_path)

    probabilities = pc.seq_probabilities[seq]
    changes = pc.seq_change[seq]
    seqs_mutated = mutate_seq_insilico(seq)

    if using_ref:
        muts = pc.results[(pc.results['result_type'] == 'New')]['ref_muts_str'].values.tolist()
        all_muts = []
        for m in muts:
            m_lst = [x.strip() for x in m.split(';')]
            all_muts.extend(m_lst)
        all_muts = list(set(all_muts))
    else:
        all_muts = []
        for parent_id, children in pc.parent_child.items():
            s = pc.tree_nodes[parent_id].spike_seq
            if s == seq:
                for child_meta in children:
                    sig_muts = child_meta['corrected_muts']
                    mut_map = child_meta['corrected_mut_map']
                    sig_muts = [x for x in sig_muts if x in mut_map and mut_map[x] in pc.ref_exp_muts]
                    all_muts.extend(sig_muts)
        all_muts = list(set(all_muts))

    seq_mutations = get_seq_mutation_dict(seq, probabilities, changes, all_muts)
    df = pd.DataFrame(seq_mutations.values())

    seq_attn = {}
    seqs_for_attn = [seq]
    seq_attn = attention_change_batchs(seqs_for_attn, bio_trans_model_ft, tokenizer, device, seq_attn,
                                            save_path=save_path, chunksize=10,
                                            pool_heads='max', pool_layer='max', l1_norm=False)
    mut_attn = seq_attn[seq]

    df['mut_l2_max_pooled'] = df['mutation'].map(mut_attn)

    if using_ref:
        save_path1 = args.save_folder+"/ref_attn_df1_l2.pkl"
    else:
        save_path1 = args.save_folder+"/parent_seq_attn_df1_l2.pkl"

    df.to_pickle(save_path1)








