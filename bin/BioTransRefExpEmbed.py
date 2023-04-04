from MutationRankingResults import *
from MutationHelpers import *
from BioTransLanguageModel import *
import argparse
from DataHelpers import chunk_dictionary, check_directory


def parse_args():
    parser = argparse.ArgumentParser(description='Biotransformer Reference Experiment')
    parser.add_argument('--data_folder', type=str, help='Folder to find related data')
    parser.add_argument('--model_type', type=str, help='Type of biotransformer model', default='protbert')
    parser.add_argument('--embed-batch', type=int, default=500,
                        help='Embedding Combination Batch size')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    args = parse_args()

    #ref_seq = load_ref_spike()
    #ref_seq = str(ref_seq.seq)

    with open("data/processed/ncbi_tree_v1/NSCS_sub_v1/embed/parent1/parent_seq1.pkl", 'rb') as f:
        ref_seq = pickle.load(f)

    exp_folder = args.data_folder + "/exp_settings"

    file = open(exp_folder + "/model_folder.txt", "r")
    model_folder = file.read()
    file.close()

    model_path = last_biotrans_ft_path(model_folder)
    bio_trans = load_biotrans(model_path=model_path)

    seqs_mutated = mutate_seq_insilico(ref_seq)
    seqs_mutated = {k:seqs_mutated[k]['mutation'] for k in seqs_mutated}

    embed_save_folder = args.data_folder + "/embed/parent1/"
    check_directory(embed_save_folder)

    ref_seqs = [str(ref_seq)]

    ref_embeddings = bio_trans.compute_embeddings(ref_seqs,  batch_size=40)
    ref_embedding = ref_embeddings['full'][0]

    save_name = embed_save_folder + "parent1_embed.pkl"
    with open(save_name, 'wb') as f:
        pickle.dump(ref_embedding, f)


    seqs_mutated_chunked = chunk_dictionary(seqs_mutated, 5000)

    for i, seq_mutated_chunk in enumerate(seqs_mutated_chunked):
        save_name = embed_save_folder + 'parent1_mut_embed_part_{}.pkl'.format(i)
        seqs = list(seq_mutated_chunk.keys())
        mutation_names = list(seq_mutated_chunk.values())
        seqs_embeddings = bio_trans.compute_embeddings(seqs, batch_size=40)
        seqs_embeddings = seqs_embeddings['full']
        seqs_embeddings = dict(zip(mutation_names, seqs_embeddings))
        with open(save_name, 'wb') as f:
            pickle.dump(seqs_embeddings, f)


    print('Done with embedding')





