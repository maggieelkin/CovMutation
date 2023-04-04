from MutationRankingResults import *
from MutationHelpers import *
from BioTransLanguageModel import *
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Biotransformer Reference Experiment')
    parser.add_argument('--log_folder', type=str, help='Folder of finetuned biotrans model')
    parser.add_argument('--all_seqs', action='store_true',
                        help='Set Test sequences to all')
    parser.add_argument('--model_type', type=str, help='Type of biotransformer model', default='protbert')
    parser.add_argument('--embed-batch', type=int, default=400,
                        help='Embedding Combination Batch size')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    ref_seq = load_ref_spike()
    ref_seq = str(ref_seq.seq)

    with open('data/processed/full_data/train_test_seq.pkl', 'rb') as f:
        train_test = pickle.load(f)

    if args.all_seqs:
        test_seqs = list(train_test['test'])
        train_seqs = list(train_test['train'])
        test_seqs = test_seqs + train_seqs
    else:
        test_seqs = list(train_test['test'])

    print('Number of test sequences: {}'.format(len(test_seqs)))
    
    # significant mutations from dataset
    muts = {'K417E', 'K444Q', 'V445A', 'N450D', 'Y453F', 'L455F', 'E484K', 'G485D', 'F486V', 'F490L',
            'F490S', 'Q493K', 'H655Y', 'R682Q', 'R685S', 'V687G', 'G769E', 'Q779K', 'V1128A'}

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V',
    ]

    seqs_mutated = mutate_seq_insilico(ref_seq, significant_mutations=muts)

    ref_seqs = [str(ref_seq)]
    log_folder = args.log_folder
    model_path = last_biotrans_ft_path(log_folder)
    bio_trans = load_biotrans(model_path=model_path)

    acc_after = bio_trans.compute_accuracy(ref_seqs, batch_size=10)
    print(f"Accuracy after finetuning : {acc_after}")
    print()

    acc_after_test = bio_trans.compute_accuracy(test_seqs, batch_size=20)
    print(f"Test Set Accuracy after finetuning : {acc_after_test}")
    print()

    print('getting masked probabilities')

    probabilities = bio_trans.compute_probabilities(ref_seqs, pass_mode='masked', batch_size=20)
    probabilities = probabilities[0]

    seqs_mutated = get_mutation_probabilities(seq=ref_seq, probabilities=probabilities, seqs_mutated_dict=seqs_mutated,
                                              prob_col='masked_prob')

    print('getting forward probabilities')
    forward_probabilities = bio_trans.compute_probabilities(ref_seqs, pass_mode='forward', batch_size=20)
    forward_probabilities = forward_probabilities[0]

    seqs_mutated = get_mutation_probabilities(seq=ref_seq, probabilities=forward_probabilities,
                                              seqs_mutated_dict=seqs_mutated, prob_col='prob')

    print('done with probabilities')
    print()
    print('embeddings')

    ref_embeddings = bio_trans.compute_embeddings(ref_seqs,  batch_size=40)
    ref_embedding = ref_embeddings['full'][0]
    seqs = list(seqs_mutated.keys())

    comb_batch = args.embed_batch
    n_batches = math.ceil(float(len(seqs)) / comb_batch)
    print('total batches: ', str(n_batches))

    for batchi in range(n_batches):
        print('Batch #', str(batchi))
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        subseqs = seqs[start:end]
        seqs_embeddings = bio_trans.compute_embeddings(subseqs, batch_size=40)
        seqs_embeddings = seqs_embeddings['full']
        for i, embedding in enumerate(seqs_embeddings):
            seq = subseqs[i]
            sem_change = abs(ref_embedding - embedding).sum()
            meta = seqs_mutated[seq]
            meta['change'] = sem_change
        del seqs_embeddings

    print('Done with embedding')

    df = pd.DataFrame()
    for seq in seqs:
        df_row = pd.DataFrame.from_dict(seqs_mutated[seq], orient='index').T
        df = df.append(df_row)

    df = df.reset_index(drop=True)

    save_name = log_folder.split('/')[1]
    save_name = '{}.pkl'.format(save_name)

    df.to_pickle(save_name)
    print('Done')

    print(df.head())

    print()

    cscs_auc, change_auc, prob_auc = cscs(df, 'significant', plot=False, flip_change_rank=True, prob_col='masked_prob')

    print('CSCS AUC:' + str(cscs_auc))
    print('Change AUC:' + str(change_auc))
    print('Prob AUC:' + str(prob_auc))




