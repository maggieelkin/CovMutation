from DataHelpers import *
from ParentChildMutate import *
from PhyloTreeParsers import *
from MutationRankingResults import cscs
from scipy.spatial import distance
import argparse


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
    arguments = parser.parse_args()
    return arguments


def get_results(sig_thres, df, prob_col='prob', change_col='change'):
    results = pd.DataFrame()
    i = 0
    for sig_muts_str, threshold in sig_thres.items():
        sig_muts = [x.strip() for x in sig_muts_str.split(';')]
        df['sig'] = False
        df['sig'] = df['mutation'].isin(sig_muts)
        n_pos = sum(df['sig'])
        cscs_auc, change_auc, prob_auc = cscs(df, 'sig', prob_col, change_col)
        results_row = pd.DataFrame({
            'sig_muts': sig_muts_str,
            'threshold': threshold,
            'n_pos': n_pos,
            'cscs_auc': cscs_auc,
            'change_auc': change_auc,
            'prob_auc': prob_auc
        }, index=[i])
        results = results.append(results_row)
        i = i + 1
    return results


def get_results_summary(data):
    groupby_col = ['threshold', 'sig_muts', 'n_pos', 'cscs_auc', 'change_auc', 'prob_auc']
    groupby_col = [x for x in groupby_col if x in data]
    agg_cols = ['cscs_auc', 'change_auc', 'prob_auc']
    agg_dict = {}
    agg_dict.update({x: 'mean' for x in agg_cols if x in data})
    groupby_col = [x for x in groupby_col if x not in agg_dict]
    data = data.groupby(groupby_col).agg(agg_dict).reset_index()
    data['threshold'] = data['threshold'].astype(int)
    thresholds = sorted(list(set(data['threshold'])))
    agg_dict = {'n_pos': 'mean',
                'cscs_auc': 'mean', 'change_auc': 'mean', 'prob_auc': 'mean',
                }
    agg_dict = {k: agg_dict[k] for k in list(agg_dict.keys()) if k in data}
    summary = pd.DataFrame()
    for threshold in thresholds:
        df1 = pd.DataFrame(data[(data['threshold'] >= threshold)].agg(agg_dict)).T
        # df1 = rename_multicol_df(df1)
        df1.insert(1, 'threshold', threshold)
        summary = summary.append(df1)
    return summary


def sliding_local_change(mut, ref_embed, mut_embed, window_size=7):
    wt, pos, alt = pullout_pos_mut(mut)
    local_changes = []
    max_n = ref_embed.shape[0]
    idx_starts = range(pos - window_size, pos + 2)
    for i, idx_start in enumerate(idx_starts):
        idx_1 = max([0, idx_start])
        idx_stop = pos + i
        idx_2 = min([max_n, idx_stop])
        local_change = abs(ref_embed[idx_1:idx_2] - mut_embed[idx_1:idx_2]).sum()
        local_changes.append(local_change)
    avg_local_change = np.mean(local_changes)
    return avg_local_change


def calc_local_change(mut, ref_embed, mut_embed, window_size=7):
    wt, pos, alt = pullout_pos_mut(mut)
    diff = int(window_size / 2)
    max_n = ref_embed.shape[0]
    idx_1 = max([0, pos - diff])
    idx_2 = min(max_n, pos + diff)
    local_change = abs(ref_embed[idx_1:idx_2] - mut_embed[idx_1:idx_2]).sum()
    return local_change


def calc_change(ref_array, mut_dict, change_type):
    mut_changes = {}
    for mut, mut_array in tqdm(mut_dict.items(), desc='Calculating {}'.format(change_type)):
        if change_type == 'l1':
            change = abs(ref_array - mut_array).sum()
        else:
            change = np.sqrt(np.sum((ref_array - mut_array) ** 2))
        mut_changes[mut] = change
    return mut_changes



def calc_agg_change(ref_array, mut_dict, change_type, agg_type):
    if agg_type == 'mean':
        ref_val = ref_array.mean(1)
    else:
        ref_val = ref_array.max(1)
    mut_changes = {}
    for mut, mut_array in tqdm(mut_dict.items(), desc='Calculating {}'.format(change_type)):
        if agg_type == 'mean':
            mut_val = mut_array.mean(1)
        else:
            mut_val = mut_array.max(1)
        if change_type == 'cosine':
            change = distance.cosine(ref_val, mut_val)
        else:
            change = distance.euclidean(ref_val, mut_val)
        mut_changes[mut] = change
    return mut_changes



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

    folder = args.data_folder + "/attn/parent/"

    pc.run_experiment(include_change=True, load_previous=True, excel=False)

    with open(folder+"parent_seq.pkl", 'rb') as f:
        parent_seq = pickle.load(f)


    all_muts = []

    for parent_id, children in pc.parent_child.items():
        seq = pc.tree_nodes[parent_id].spike_seq
        if seq == parent_seq:
            for child_meta in children:
                sig_muts = child_meta['corrected_muts']
                mut_map = child_meta['corrected_mut_map']
                sig_muts = [x for x in sig_muts if x in mut_map and mut_map[x] in pc.ref_exp_muts]
                all_muts.extend(sig_muts)

    all_muts = list(set(all_muts))

    probabilities = pc.seq_probabilities[parent_seq]
    changes = pc.seq_change[parent_seq]
    seqs_mutated = mutate_seq_insilico(parent_seq)
    seq_mutations = get_seq_mutation_dict(parent_seq, probabilities, changes, all_muts)
    df = pd.DataFrame(seq_mutations.values())

    df = df.rename(columns={'change': 'Semantic Change'})

    with open(folder + 'parent_mean_attn.pkl', 'rb') as f:
        ref_attn = pickle.load(f)

    ref_mut_attn = {}
    for file in os.listdir(folder):
        if file == "parent_mean_attn.pkl" or file == "parent_seq.pkl":
            continue
        p = folder + file
        with open(p, 'rb') as f:
            ref_mut_attn_part = pickle.load(f)
            ref_mut_attn.update(ref_mut_attn_part)

    del ref_mut_attn_part

    l1_attn_change = calc_change(ref_array=ref_attn, mut_dict=ref_mut_attn, change_type='l1')

    df["L1 Attn Change"] = df['mutation'].map(l1_attn_change)


    l2_attn_change = calc_change(ref_array=ref_attn, mut_dict=ref_mut_attn, change_type='l2')

    df["L2 Attn Change"] = df['mutation'].map(l2_attn_change)


    cos_mean_change = calc_agg_change(ref_array=ref_attn, mut_dict=ref_mut_attn,
                                      change_type='cosine', agg_type='mean')

    df["Cos Mean Attn Change"] = df['mutation'].map(cos_mean_change)


    cos_max_change = calc_agg_change(ref_array=ref_attn, mut_dict=ref_mut_attn,
                                     change_type='cosine',  agg_type='max')

    df["Cos Max Attn Change"] = df['mutation'].map(cos_max_change)


    euc_mean_change = calc_agg_change(ref_array=ref_attn, mut_dict=ref_mut_attn,
                                      change_type='euclidean', agg_type='mean')

    df["Euc Mean Attn Change"] = df['mutation'].map(euc_mean_change)


    euc_max_change = calc_agg_change(ref_array=ref_attn, mut_dict=ref_mut_attn,
                                     change_type='euclidean', agg_type='max')
    df["Euc MAx Attn Change"] = df['mutation'].map(euc_max_change)


    df.to_pickle("parent_attn_df.pkl")

    with open('ParentAttnDF.pkl', 'wb') as f:
        pickle.dump(df, f)

    print('Done')




