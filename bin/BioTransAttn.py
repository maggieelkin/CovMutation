from transformers import BertForMaskedLM, BertTokenizer
from biotransformers.lightning_utils.data import AlphabetDataLoader
from biotransformers.lightning_utils.models import LightningModule
from ParentChildMutate import *
from scipy import special
import copy

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


def get_alphabet_dataloader(tokenizer, model_dir):
    """Define an alphabet mapping for common method between
    protbert and ESM
    """

    def tokenize(x):
        x_ = [" ".join(seq) for seq in x]
        tokens = tokenizer(x_, return_tensors="pt", padding=True)
        return x, tokens["input_ids"]

    all_tokens = copy.deepcopy(tokenizer.vocab)
    del all_tokens["[PAD]"]
    del all_tokens["[UNK]"]
    del all_tokens["[CLS]"]
    del all_tokens["[SEP]"]
    del all_tokens["[MASK]"]
    standard_tokens = list(all_tokens.keys())

    alphabet_dl = AlphabetDataLoader(
        prepend_bos=True,
        append_eos=True,
        mask_idx=tokenizer.mask_token_id,
        pad_idx=tokenizer.pad_token_id,
        standard_toks=standard_tokens,
        model_dir=model_dir,
        lambda_toks_to_ids=lambda x: tokenizer.convert_tokens_to_ids(x),
        lambda_tokenizer=lambda x: tokenize(x),
    )
    return alphabet_dl


def get_attention(seq, tokenizer, model, device, pool_heads=None):
    seq_space = (" ").join(seq)
    inputs_seq = tokenizer.encode_plus(seq_space, return_tensors='pt', add_special_tokens=True)
    input_seq_ids = inputs_seq['input_ids']
    attention = model(input_seq_ids.to(device))[-1]
    layer_attentions = format_attention(attention)
    layer_attentions = layer_attentions.cpu().detach().numpy()
    if pool_heads=='mean':
        layer_attentions = layer_attentions.mean(axis=1)
    elif pool_heads=='max':
        layer_attentions = layer_attentions.max(axis=1)
    return layer_attentions


def pool_attention(layer_attentions, pool_layer='max'):
    if pool_layer == 'max':
        attn = layer_attentions.max(axis=0)
    else:
        attn = layer_attentions.mean(axis=0)
    return attn


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


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_dir = "Rostlab/prot_bert"

    tokenizer = BertTokenizer.from_pretrained(
        model_dir, do_lower_case=False, padding=True
    )

    alphabet_dl = get_alphabet_dataloader(tokenizer, model_dir)

    module = LightningModule.load_from_checkpoint(
        checkpoint_path=model_path,
        model=BertForMaskedLM.from_pretrained(model_dir, output_attentions=True).eval().to(device),
        alphabet=alphabet_dl,
        lr=1e-5,
        warmup_updates=100,
        warmup_init_lr=1e-7,
        warmup_end_lr=1e-5,
    )
    bio_trans_model_ft = module.model.to(device)

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

    ref_attn = get_attention(seq, tokenizer, bio_trans_model_ft, device, pool_heads='max')
    ref_max_attn = pool_attention(ref_attn, pool_layer='max')
    #ref_max_attn = normalize_attn_mat(ref_max_attn)
    #ref_mean_attn = pool_attention(ref_attn, pool_layer='mean')
    #ref_mean_attn = normalize_attn_mat(ref_mean_attn)

    mutation_seqs = {seqs_mutated[s]['mutation']: s for s in list(seqs_mutated.keys())}

    #mut_kl_max_pooled = {}
    #mut_kl_mean_pooled = {}
    mut_max_pooled = {}
    for mut, mut_seq in tqdm(mutation_seqs.items(), desc="Mutation Attention"):
        mut_attn = get_attention(mut_seq, tokenizer, bio_trans_model_ft, device, pool_heads='max')
        mut_max_attn = pool_attention(mut_attn, pool_layer='max')
        #mut_max_attn = normalize_attn_mat(mut_max_attn)
        #kl_max = kl_divergence(ref_max_attn, mut_max_attn)
        #mut_kl_max_pooled[mut] = kl_max
        #mut_mean_attn = pool_attention(mut_attn, pool_layer='mean')
        #mut_mean_attn = normalize_attn_mat(mut_mean_attn)
        #kl_mean = kl_divergence(ref_mean_attn, mut_mean_attn)
        #mut_kl_mean_pooled[mut] = kl_mean
        l2_change = np.sqrt(np.sum((ref_max_attn - mut_max_attn) ** 2))
        mut_max_pooled[mut] = l2_change

    #df['mut_kl_max_pooled'] = df['mutation'].map(mut_kl_max_pooled)
    #df['mut_kl_mean_pooled'] = df['mutation'].map(mut_kl_mean_pooled)
    df['mut_l2_max_pooled'] = df['mutation'].map(mut_max_pooled)

    if using_ref:
        save_path = args.save_folder+"/ref_attn_df_l2.pkl"
    else:
        save_path = args.save_folder+"/parent_seq_attn_df_l2.pkl"

    df.to_pickle(save_path)








