from ParentChildMutate import *
from BioTransLanguageModel import *


def parse_args():
    parser = argparse.ArgumentParser(description='Fake ParentChild Mutate experiment')
    parser.add_argument('--tree_version', type=int, help='Tree Version for models')
    parser.add_argument('--finetuned', action='store_true',
                        help='to use fine tuned biotransformer model')
    parser.add_argument('--masked_mode', action='store_false',
                        help='if provided, experiment will use masked mode')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
    parser.add_argument('--include_change', action='store_true',
                        help='include change in experiment')
    parser.add_argument('--list_seq_batchsize', type=int, default=15,
                        help='number of sequences to calculate embedding change/probabilities at a time')
    parser.add_argument('--prob_batchsize', type=int, default=20,
                        help='token batchsize for calculating prob')
    parser.add_argument('--seq_batchsize', type=int, default=500,
                        help='sequence list batch for embedding')
    parser.add_argument('--embedding_batchsize', type=int, default=50,
                        help='mini sequence batchsize for embedding')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    pc_exp = MutateParentChild(tree_version=args.tree_version, finetuned=args.finetuned, forward_mode=args.masked_mode,
                               data_folder=args.data_folder)
    print("masked mode arg: {}".format(args.masked_mode))
    print("Finetuned is: {}".format(pc_exp.finetuned))
    print("Forward Mode is: {}".format(pc_exp.forward_mode))

    save_prob_path = pc_exp.data_folder + '/fake_seqs_proba.pkl'

    with open(pc_exp.data_folder + '/fake_seqs_for_proba.pkl', 'rb') as f:
        seqs_for_proba = pickle.load(f)
    seqs_for_proba, seq_probabilities = find_previous_saved(seqs_for_proba, save_prob_path)

    bio_trans = load_biotrans(model_path=pc_exp.model_path)
    print('computing probabilities for {} sequences'.format(len(seqs_for_proba)))

    seq_probabilities = compute_probabilities(bio_trans=bio_trans, seqs=seqs_for_proba,
                                              batchsize=args.prob_batchsize, chunksize=args.list_seq_batchsize,
                                              forward_mode=False,
                                              prior_seq_probabilities=seq_probabilities,
                                              save_path=save_prob_path)

    print('Saving {} sequence probabilities'.format(len(seq_probabilities)))
    with open(save_prob_path, 'wb') as a:
        pickle.dump(seq_probabilities, a)
