from DataHelpers import check_directory
from MutationRankingResults import *
from MutationHelpers import *
from BioTransLanguageModel import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Biotransformer Reference Experiment')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
    parser.add_argument('--prob_mode', type=str, default='forward',
                        help='"masked" for masked mode, otherwise "forward" for forward mode')
    parser.add_argument('--norm_mode', type=str, default='l2',
                        help='"l1" for l1 norm change, otherwise "l2" for l2 norm change')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save calculated values to.')
    parser.add_argument('--seq_batchsize', type=int, default=400,
                        help='sequence list batch for embedding')
    parser.add_argument('--embedding_batchsize', type=int, default=40,
                        help='mini sequence batchsize for embedding')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    print(args)

    ref_seq = load_ref_spike()
    ref_seq = str(ref_seq.seq)

    exp_folder = args.data_folder + '/exp_settings'
    file = open(exp_folder + "/model_folder.txt", "r")
    model_folder = file.read()
    file.close()

    mutation_data_path = exp_folder + '/mutation_data.pkl'
    mutation_data = pd.read_pickle(mutation_data_path)
    mutation_data = mutation_data[(mutation_data['test_set_mut'])]
    sig_thres = pd.Series(mutation_data.n_times.values, index=mutation_data.mutation).to_dict()
    sig_muts = list(sig_thres.keys())

    seqs_mutated = mutate_seq_insilico(ref_seq, significant_mutations=sig_muts)

    ref_seqs = [str(ref_seq)]

    model_path = last_biotrans_ft_path(model_folder)
    bio_trans = load_biotrans(model_path=model_path)

    if args.prob_mode == 'forward':
        print('using forward mode')
        probabilities = bio_trans.compute_probabilities(ref_seqs, pass_mode='forward', batch_size=20)
    else:
        print('using masked mode')
        probabilities = bio_trans.compute_probabilities(ref_seqs, pass_mode='masked', batch_size=20)

    probabilities = probabilities[0]

    # seqs_mutated = get_mutation_probabilities(seq=ref_seq, probabilities=probabilities, seqs_mutated_dict=seqs_mutated,
    #                                           prob_col='prob')

    print('done with probabilities')
    print()
    print('embeddings')

    if args.norm_mode == 'l1':
        changes = get_mutation_embedding_change(ref_seq, bio_trans, seq_batchsize=args.seq_batchsize,
                                                embedding_batchsize=args.embedding_batchsize, l1_norm=True)
    else:
        changes = get_mutation_embedding_change(ref_seq, bio_trans, seq_batchsize=args.seq_batchsize,
                                                embedding_batchsize=args.embedding_batchsize, l1_norm=False)

    print('Done with embedding')

    del bio_trans

    # custom bio_transformer is built to calculate attention
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, bio_trans = load_biotrans_for_attn(device=device, model_path=model_path)

    attn = get_mutation_attention_change(seq_to_mutate=ref_seq, bio_trans=bio_trans, tokenizer=tokenizer,
                                         device=device, pool_heads_max=True,
                                         pool_layer_max=True, l1_norm=False)

    seq_mutations = get_seq_mutation_dict(seq=ref_seq, probabilities=probabilities, changes=changes,
                                          significant_mutations=sig_muts, attn=attn)

    ranking_values = {'alpha': 1.5, 'beta': 3.0, 'gamma': 1.0}

    results = results_over_thresholds(seq_mutations, sig_muts, 'ref_seq', sig_thres, mut_map=None, parent_in_train=None,
                                      **ranking_values)

    #print("Average per Mutation")
    #measures = ['prob_auc', 'change_auc', 'attn_auc', 'cscs_auc', 'dnms_auc']
    #avg = round(results.groupby('result_type')[measures].mean(), 4)
    #print(avg)
    #print()

    ref_folder = args.data_folder + '/ref'

    check_directory(ref_folder)

    df = pd.DataFrame(seq_mutations.values())

    df.to_pickle(ref_folder+"/ref_muts.pkl")
    results.to_pickle(ref_folder+'ref_results.pkl')



