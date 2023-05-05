from BioTransTree import *
from BioTransLanguageModel import *
import time

class BioTransTreeAttn(BioTransTree):
    """

    """

    def __init__(self, tree_version, data_folder, finetuned=True, model_folder=None):
        super().__init__(tree_version, data_folder, finetuned, forward_mode=False, model_folder=model_folder,
                         cut_off='2022-1-1')
        self.seq_attn = {}

    def runnit_attn(self, all_seqs=True, start_idx=0, n_seqs=500, pool_heads_max=True, pool_layer_max=True):
        self.manage_attn_save_files()
        self.get_attn_leaf_seqs()
        self.calc_attn(all_seqs, start_idx, n_seqs, pool_heads_max=pool_heads_max, pool_layer_max=pool_layer_max)

    def manage_attn_save_files(self):
        if self.finetuned:
            train_str = 'ft'
        else:
            train_str = 'pretrain'
        suffix = 'tree_v{}_seq_attn_{}.pkl'.format(self.tree_version, train_str)
        self.full_save_path = self.data_folder + "/leaf_{}".format(suffix)

    def get_attn_leaf_seqs(self):
        seqs = []
        for node_id, node in self.tree_nodes.items():
            if node.leaf:
                seqs.append(node.spike_seq)
        self.all_leaf_seqs = list(set(seqs))
        print("Total Leaf Sequences: {}".format(len(self.all_leaf_seqs)))
        self.leaf_seqs = self.all_leaf_seqs
        self.leaf_seqs, self.seq_attn = find_previous_saved(self.leaf_seqs, self.full_save_path, print_msg=False)
        print("Total Leaf Sequences Left to calculate attention: {}".format(len(self.leaf_seqs)))

    def calc_attn(self, all_seqs=True, start_idx=0, n_seqs=500, pool_heads_max=True, pool_layer_max=True):
        if all_seqs:
            print("Calculating for all sequences")
            seqs_for_attn = self.leaf_seqs
        else:
            end_idx = start_idx + n_seqs
            seqs_for_attn = self.leaf_seqs[start_idx: end_idx]
        seqs_for_attn = {k: v for k, v in enumerate(seqs_for_attn)}
        ray.init()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        tokenizer, bio_trans = load_biotrans_for_attn(device=device, model_path=self.model_path)
        model_ref = ray.put(bio_trans)
        n_gpu = torch.cuda.device_count()
        actor_pool = [BertActor.remote(model=model_ref, device=device, tokenizer=tokenizer) for _ in range(n_gpu)]
        start = time.time()
        results = []
        for index, seq in seqs_for_attn.items():
            actor = actor_pool.pop(0)
            results.append(actor.predict_attention.remote(mut=index, seq=seq, pool_heads_max=pool_heads_max,
                                                          pool_layer_max=pool_layer_max))
            actor_pool.append(actor)  # Return the actor to the pool

        result_dict = {}
        pbar = tqdm(total=len(results))
        while len(results) > 0:
            done_ids, results = ray.wait(results)
            for done_id in done_ids:
                result_dict = process_incremental_mean_attn(results=result_dict, attn_dict=ray.get(done_id),
                                                              index_seq_dict=seqs_for_attn)
                pbar.update(1)
        pbar.close()
        ray.shutdown()
        end = time.time()
        print("Time to run: {}".format(end-start))
        self.seq_attn.update(result_dict)
        print('Saving {} sequence Attentions'.format(len(self.seq_attn)))
        with open(self.full_save_path, 'wb') as a:
            pickle.dump(self.seq_attn, a)


def parse_args():
    parser = argparse.ArgumentParser(description='Biotrans Tree Sequence Attention experiment')
    parser.add_argument('--tree_version', type=int, help='Tree Version for models')
    parser.add_argument('--finetuned', action='store_true',
                        help='to use fine tuned biotransformer model')
    parser.add_argument('--data_folder', type=str,
                        help='Folder for data to use in experiment')
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

    biotranstree_exp = BioTransTreeAttn(tree_version=args.tree_version, finetuned=args.finetuned,
                                        data_folder=args.data_folder)

    biotranstree_exp.runnit_attn(all_seqs=args.all_seqs, n_seqs=args.n_seqs,
                                  start_idx=args.start_idx)


