from ParentChildMutate import *
from scipy import special
from BioTransLanguageModel import *
import time

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
    parser.add_argument('--attn_seq_batchsize', type=int, default=200,
                        help='sequence list batch for Attention')
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
    '''
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
    '''
    if args.seq_path is None:
        seq_path = 'data/processed/ncbi_tree_v1/NSCS_sub_v1/attn/parent/parent_seq.pkl'
    else:
        seq_path = args.seq_path

    seqs = []
    ref_seq = load_ref_spike()
    ref_seq = str(ref_seq.seq)
    seqs.append(ref_seq)
    with open(args.seq_path, 'rb') as f:
        parent_seq = pickle.load(f)
    seqs.append(parent_seq)

    exp_folder = args.data_folder + "/exp_settings"
    file = open(exp_folder + "/model_folder.txt", "r")
    model_folder = file.read()
    file.close()

    model_path = last_biotrans_ft_path(model_folder)

    save_path = args.save_folder+"/tree_v1_seq_attn_ft.pkl"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, bio_trans = load_biotrans_for_attn(device=device, model_path=model_path)

    seq_attn = {}
    start = time.time()
    seq_attn = attention_change_batchs(seqs, bio_trans, tokenizer, device, seq_attn=seq_attn,
                                       seq_batchsize=args.attn_seq_batchsize,
                                       save_path=save_path, pool_heads_max=True, pool_layer_max=True,
                                       l1_norm=False)
    end = time.time()
    diff = end - start
    print("Time to run 2 seqs: {}".format(diff))

    print('Saving {} sequence attention change'.format(len(seq_attn)))
    with open(save_path, 'wb') as a:
        pickle.dump(seq_attn, a)















