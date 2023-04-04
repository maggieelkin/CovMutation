from BioTransTree import *
from BioTransLanguageModel import calculate_sequence_embedding


class BioTransTreeEmbed(BioTransTree):
    """

    """

    def __init__(self, tree_version, data_folder, finetuned=True, model_folder=None):
        super().__init__(tree_version, data_folder, finetuned, forward_mode=False, model_folder=model_folder,
                         cut_off='2022-1-1')
        self.seq_embed = {}

    def runnit_embed(self, all_seqs=True, start_idx=0, n_seqs=500, embedding_batchsize=40,
                     chunksize=100):
        self.manage_embed_save_files()
        self.get_embed_leaf_seqs()
        self.calc_embedding(all_seqs, start_idx, n_seqs, embedding_batchsize, chunksize)

    def manage_embed_save_files(self):
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        suffix = 'tree_v{}_seq_embed_{}.pkl'.format(self.tree_version, train_str)
        self.full_save_path = self.data_folder + "/leaf_{}".format(suffix)

    def get_embed_leaf_seqs(self):
        seqs = []
        for node_id, node in self.tree_nodes.items():
            if node.leaf:
                seqs.append(node.spike_seq)
        self.all_leaf_seqs = list(set(seqs))
        print("Total Leaf Sequences: {}".format(len(self.all_leaf_seqs)))
        self.leaf_seqs = self.all_leaf_seqs
        self.leaf_seqs, self.seq_embed = find_previous_saved(self.leaf_seqs, self.full_save_path, print_msg=False)
        print("Total Leaf Sequences Left to calculate embedding: {}".format(len(self.leaf_seqs)))

    def calc_embedding(self, all_seqs=True, start_idx=0, n_seqs=500, embedding_batchsize=40, chunksize=100):
        bio_trans = load_biotrans(model_path=self.model_path)
        if all_seqs:
            print("Calculating for all sequences")
            seqs_for_embed = self.leaf_seqs
        else:
            end_idx = start_idx + n_seqs
            seqs_for_embed = self.leaf_seqs[start_idx: end_idx]

        print('Calculating Embeddings for {} sequences'.format(len(seqs_for_embed)))
        self.seq_embed = calculate_sequence_embedding(bio_trans=bio_trans, seqs=seqs_for_embed,
                                                      embedding_batchsize=embedding_batchsize, seq_embed=self.seq_embed,
                                                      save_path=self.full_save_path, chunksize=chunksize)

        print('Saving {} sequence Embeddings'.format(len(self.seq_embed)))
        with open(self.full_save_path, 'wb') as a:
            pickle.dump(self.seq_embed, a)


def parse_args():
    parser = argparse.ArgumentParser(description='Biotrans Tree Sequence Embedding experiment')
    parser.add_argument('--tree_version', type=int, help='Tree Version for models')
    parser.add_argument('--finetuned', action='store_true',
                        help='to use fine tuned biotransformer model')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
    parser.add_argument('--list_seq_batchsize', type=int, default=100,
                        help='number of sequences to calculate embedding at a time (chunksize)')
    parser.add_argument('--embedding_batchsize', type=int, default=50,
                        help='sequence batchsize for bio_trans embedding')
    parser.add_argument('--all_seqs', action='store_true',
                        help='if provided, experiment will only calculate for all sequences')
    parser.add_argument('--n_seqs', type=int, default=500,
                        help='number of sequences to calculate probabilities for')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='index position of sequence list to start')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    biotranstree_exp = BioTransTreeEmbed(tree_version=args.tree_version, finetuned=args.finetuned,
                                         data_folder=args.data_folder)

    biotranstree_exp.runnit_embed(all_seqs=args.all_seqs, n_seqs=args.n_seqs,
                                  start_idx=args.start_idx, chunksize=args.list_seq_batchsize,
                                  embedding_batchsize=args.embedding_batchsize)
