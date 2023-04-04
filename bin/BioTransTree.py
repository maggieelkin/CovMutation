import argparse
from PhyloTreeParsers import Node
from ParentChildMutate import BioTransExpSettings
from BioTransLanguageModel import find_previous_saved, load_biotrans, compute_probabilities
from DataHelpers import check_directory
import uuid
import os
from datetime import datetime
import pickle


class BioTransTree(BioTransExpSettings):
    """

    """

    def __init__(self, tree_version, data_folder, finetuned=True, forward_mode=False, model_folder=None,
                 cut_off='2022-1-1'):
        super().__init__(tree_version, data_folder, finetuned, forward_mode, model_folder, cut_off)
        self.prob_save_folder = self.data_folder + "/probas"
        check_directory(self.prob_save_folder)
        self.str_id = uuid.uuid1()
        self.prob_save_files = []
        self.current_save_path = None
        self.full_save_path = None
        self.all_leaf_seqs = []
        self.leaf_seqs = []

    def runnit(self, postcut_only=False, all_seqs=False, n_seqs=500, start_idx=0,
               prob_batchsize=20, list_seq_batchsize=15, combine=True):
        self.manage_save_files()
        self.get_leaf_seqs(postcut_only=postcut_only)
        self.calc_probas(all_seqs, n_seqs, start_idx, prob_batchsize, list_seq_batchsize)
        if combine:
            print("Combining Proba Save Files")
            self.combine_probas()

    def manage_save_files(self):
        if self.forward_mode:
            mode_str = 'forward'
        else:
            mode_str = 'masked'
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        suffix = 'tree_v{}_seq_{}_prob_{}.pkl'.format(self.tree_version, mode_str, train_str)
        if os.path.isfile(self.seq_prob_path):
            self.prob_save_files.append(self.seq_prob_path)
        for file in os.listdir(self.prob_save_folder):
            if file.endswith(suffix):
                self.prob_save_files.append(self.prob_save_folder + '/' + file)
        self.current_save_path = self.prob_save_folder + '/{}_{}'.format(self.str_id, suffix)
        self.full_save_path = self.data_folder + "/leaf_{}".format(suffix)

    def get_leaf_seqs(self, postcut_only=False):
        cut_off_datetime = datetime.strptime(self.cut_off, '%Y-%m-%d')
        seqs = []
        for node_id, node in self.tree_nodes.items():
            if postcut_only:
                #print("Only getting Postcut sequences")
                if node.leaf and node.date >= cut_off_datetime:
                    seqs.append(node.spike_seq)
            else:
                if node.leaf:
                    seqs.append(node.spike_seq)
        self.all_leaf_seqs = list(set(seqs))
        print("Total Leaf Sequences: {}".format(len(self.all_leaf_seqs)))
        self.leaf_seqs = self.all_leaf_seqs
        self.leaf_seqs, _ = find_previous_saved(self.leaf_seqs, self.full_save_path, print_msg=False)
        for file in self.prob_save_files:
            self.leaf_seqs, probas = find_previous_saved(self.leaf_seqs, file, print_msg=False)
        print("Total Leaf Sequences Left to calculate Probas: {}".format(len(self.leaf_seqs)))

    def calc_probas(self, all_seqs=False, n_seqs=500, start_idx=0, prob_batchsize=20, list_seq_batchsize=15):
        bio_trans = load_biotrans(model_path=self.model_path)
        if all_seqs:
            print("Calculating for all sequences")
            seqs_for_proba = self.leaf_seqs
        else:
            end_idx = start_idx + n_seqs
            seqs_for_proba = self.leaf_seqs[start_idx: end_idx]
        print('Calculating Probabilities for {} sequences'.format(len(seqs_for_proba)))
        seq_probabilities = compute_probabilities(bio_trans=bio_trans, seqs=seqs_for_proba, batchsize=prob_batchsize,
                                                  chunksize=list_seq_batchsize, forward_mode=self.forward_mode,
                                                  save_path=self.current_save_path)

        print('Saving {} sequence probabilities'.format(len(seq_probabilities)))
        with open(self.current_save_path, 'wb') as a:
            pickle.dump(seq_probabilities, a)

    def combine_probas(self):
        full_seq_probabilities = {}
        self.prob_save_files.append(self.current_save_path)
        self.prob_save_files.append(self.full_save_path)
        for file in self.prob_save_files:
            if not os.path.isfile(file):
                continue
            if file == self.seq_prob_path:
                _, probas = find_previous_saved(self.all_leaf_seqs, self.seq_prob_path, print_msg=False)
                probas = dict((k, probas[k]) for k in self.all_leaf_seqs if k in probas)
            else:
                with open(file, 'rb') as f:
                    probas = pickle.load(f)
            full_seq_probabilities.update(probas)
        print('Saving {} combined sequence probabilities'.format(len(full_seq_probabilities)))
        with open(self.full_save_path, 'wb') as f:
            pickle.dump(full_seq_probabilities, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Biotrans Tree Sequence Probabilities experiment')
    parser.add_argument('--tree_version', type=int, help='Tree Version for models')
    parser.add_argument('--finetuned', action='store_true',
                        help='to use fine tuned biotransformer model')
    parser.add_argument('--masked_mode', action='store_false',
                        help='if provided, experiment will use masked mode')
    parser.add_argument('--postcut_only', action='store_true',
                        help='if provided, experiment will only consider postcut leaf nodes')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
    parser.add_argument('--list_seq_batchsize', type=int, default=5,
                        help='number of sequences to calculate embedding probabilities at a time')
    parser.add_argument('--prob_batchsize', type=int, default=20,
                        help='token batchsize for calculating prob')
    parser.add_argument('--all_seqs', action='store_true',
                        help='if provided, experiment will only calculate for all sequences')
    parser.add_argument('--n_seqs', type=int, default=500,
                        help='number of sequences to calculate probabilities for')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='index position of sequence list to start')
    parser.add_argument('--combine', action='store_true',
                        help='if provided, experiment will combine probabilities')
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    print(args)

    biotranstree_exp = BioTransTree(tree_version=args.tree_version,
                                    finetuned=args.finetuned,
                                    forward_mode=args.masked_mode,
                                    data_folder=args.data_folder)


    biotranstree_exp.runnit(postcut_only=args.postcut_only, all_seqs=args.all_seqs, n_seqs=args.n_seqs,
                            start_idx=args.start_idx, prob_batchsize=args.prob_batchsize,
                            list_seq_batchsize=args.list_seq_batchsize, combine=args.combine)


