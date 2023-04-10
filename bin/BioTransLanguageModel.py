import math
import os
import pickle
from tqdm import tqdm
from DataHelpers import chunks
from biotransformers import BioTransformers
import torch
import ray
from MutationHelpers import mutate_seq_insilico
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer
from biotransformers.lightning_utils.data import AlphabetDataLoader
from biotransformers.lightning_utils.models import LightningModule
import copy

def biotrans_probabilities(bio_trans, seqs, batchsize, forward_mode=False):
    """

    :param batchsize: batchsize for compute_probabilities
    :type batchsize: int
    :param bio_trans: biotransformer model
    :type bio_trans:
    :param seqs: list of sequences to compute probabiltiies for
    :type seqs: list
    :param forward_mode: if true, do forward mode for biotrans
    :type forward_mode: bool
    :return: dictionary of key = sequence, value = predicted probabilities
    :rtype: dict
    :return:
    :rtype:
    """
    if forward_mode:
        probabilities = bio_trans.compute_probabilities(seqs, pass_mode='forward', batch_size=batchsize)
    else:
        probabilities = bio_trans.compute_probabilities(seqs, pass_mode='masked', batch_size=batchsize)
    seq_probabilities = dict(zip(seqs, probabilities))
    return seq_probabilities


def compute_probabilities(bio_trans, seqs,  batchsize, chunksize=None, forward_mode=False, save_path=None,
                          prior_seq_probabilities=None):
    """
    function to compute probabilities for a list of sequences
    :param prior_seq_probabilities: starting dictionary of seq_probabilities
    :type prior_seq_probabilities: dict
    :param save_path: path to save chunks of seq_probabilities
    :type save_path:
    :param chunksize: if provided, cut the list of sequences to chunks (sized by chunksize)
    :type chunksize: int
    :param batchsize: batchsize for compute_probabilities
    :type batchsize: int
    :param bio_trans: biotransformer model
    :type bio_trans:
    :param seqs: list of sequences to compute probabiltiies for
    :type seqs: list
    :param forward_mode: if true, do forward mode for biotrans
    :type forward_mode: bool
    :return: dictionary of key = sequence, value = predicted probabilities
    :rtype: dict
    """
    if prior_seq_probabilities is not None:
        seq_probabilities = prior_seq_probabilities
    else:
        seq_probabilities = {}
    if chunksize is not None:
        seqs_chunked = list(chunks(seqs, chunksize))
    else:
        seqs_chunked = [seqs]
    for seq_chunk in tqdm(seqs_chunked, desc='Chunked sequence list for probabilities'):
        seq_chunk_prob = biotrans_probabilities(bio_trans=bio_trans, seqs=seq_chunk, batchsize=batchsize,
                                                forward_mode=forward_mode)
        seq_probabilities.update(seq_chunk_prob)
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(seq_probabilities, f)
    return seq_probabilities


def load_biotrans(model_path=None):
    """

    :param model_path:
    :type model_path:
    :return:
    :rtype:
    """
    n_gpu = torch.cuda.device_count()
    ray.init()
    bio_trans = BioTransformers(backend="protbert", num_gpus=n_gpu)
    if model_path is not None:
        print("Loading Model from: {}".format(model_path))
        bio_trans.load_model(model_path)
    return bio_trans


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


def load_biotrans_for_attn(device, model_path=None):
    """

    :param device:
    :type device:
    :param model_path:
    :type model_path:
    :return:
    :rtype:
    """
    model_dir = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(
        model_dir, do_lower_case=False, padding=True
    )
    alphabet_dl = get_alphabet_dataloader(tokenizer, model_dir)
    if model_path is not None:
        module = LightningModule.load_from_checkpoint(
            checkpoint_path=model_path,
            model=BertForMaskedLM.from_pretrained(model_dir, output_attentions=True).eval().to(device),
            alphabet=alphabet_dl,
            lr=1e-5,
            warmup_updates=100,
            warmup_init_lr=1e-7,
            warmup_end_lr=1e-5,
            )
        bio_trans = module.model.to(device)
    else:
        bio_trans = BertForMaskedLM.from_pretrained(model_dir, output_attentions=True).eval().to(device)
    return tokenizer, bio_trans


def format_attention(attention, layers=None, heads=None):
    """

    :param attention:
    :type attention:
    :param layers:
    :type layers:
    :param heads:
    :type heads:
    :return:
    :rtype:
    """
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


def get_attention(seq, tokenizer, model, device, pool_heads=None):
    """

    :param seq:
    :type seq:
    :param tokenizer:
    :type tokenizer:
    :param model:
    :type model:
    :param device:
    :type device:
    :param pool_heads:
    :type pool_heads:
    :return:
    :rtype:
    """
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
    """

    :param layer_attentions:
    :type layer_attentions:
    :param pool_layer:
    :type pool_layer:
    :return:
    :rtype:
    """
    if pool_layer == 'max':
        attn = layer_attentions.max(axis=0)
    else:
        attn = layer_attentions.mean(axis=0)
    return attn


def find_previous_saved(seqs, save_path, print_msg=True):
    """
    looks for saved dictionary under save_path
    checks if the list of seqs are found in the loaded dictionary (or if it exists)

    :param print_msg: if true, print to screen about saved files
    :type print_msg: bool
    :param save_path: path to load saved dictionary
    :type save_path: str
    :param seqs: list of sequences to check
    :type seqs: list
    :return: list of sequences to run through biotransformer, loaded saved dictionary
    :rtype: tuple
    """
    seqs_for_biotrans = []
    if os.path.isfile(save_path):
        if print_msg:
            print('previous saved values found')
            print(save_path)
        with open(save_path, 'rb') as f:
            seq_values = pickle.load(f)
    else:
        if print_msg:
            print('no previous saved values')
        seq_values = {}
    for seq in seqs:
        if seq not in seq_values:
            seqs_for_biotrans.append(seq)
    return seqs_for_biotrans, seq_values


def last_biotrans_ft_path(log_folder):
    """
    from the log_folder, get the max version subfolder
    :param log_folder: subfolder in logs that holds the biotrans finetuned models
    :type log_folder: str
    :return: path to biotrans model
    :rtype: str
    """
    subfolders = []
    for subfolder in os.listdir(log_folder):
        subfolders.append(subfolder)
    max_folder_num = max([int(x.split('_')[1]) for x in subfolders])
    subfolder = [x for x in subfolders if x.endswith(str(max_folder_num))][0]
    model_path = log_folder + '/' + subfolder + '/checkpoints'
    model_filename = [x for x in os.listdir(model_path) if x.endswith('.ckpt')][0]
    model_path = model_path + '/' + model_filename
    return model_path


def calculate_sequence_embedding(seqs, bio_trans, embedding_batchsize, seq_embed=None, save_path=None,
                            chunksize=100, return_full=False):
    """
    function to calculate embedding (not change, just embedding of sequences)

    :param return_full: if true, return the full seq embedding (embed. vector for each AA) otherwise return pooled embed
    :type return_full: bool
    :param seqs: sequences to get embeddings for
    :type seqs: list
    :param bio_trans: biotrans model
    :type bio_trans: biotransformers.wrappers.transformers_wrappers.TransformersWrapper
    :param embedding_batchsize: batchsize for embedding
    :type embedding_batchsize: int
    :param seq_embed: dictionary of prior saved values, key = seq, value = embedding
    :type seq_embed: dict
    :param save_path: save path for seq_embed_dict
    :type save_path: str
    :param chunksize: breaks seqs into chunksize length lists
    :type chunksize: int
    :return: dictionary of key = seq, value = embedding
    :rtype: dict
    """
    if seq_embed is None:
        seq_embed_dict = {}
    else:
        seq_embed_dict = seq_embed
    if return_full:
        return_type = 'full'
    else:
        return_type = 'mean'
    seqs_chunked = list(chunks(seqs, chunksize))
    for seq_chunk in tqdm(seqs_chunked, desc='Chunked sequence list for Embedding'):
        seqs_embeddings = bio_trans.compute_embeddings(seq_chunk, batch_size=embedding_batchsize)
        seqs_embeddings = seqs_embeddings[return_type]
        seq_embed_dict.update(dict(zip(seq_chunk, seqs_embeddings)))
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(seq_embed_dict, f)
    return seq_embed_dict


def get_mutation_embedding_change(seq_to_mutate, bio_trans, seq_batchsize, embedding_batchsize, l1_norm=True):
    """

    :param l1_norm: if true, change is based on l1_norm, does l2_norm otherwise. Default is true
    :type l1_norm: bool
    :param embedding_batchsize: batchsize to use in compute_embeddings
    :type embedding_batchsize: int
    :param seq_batchsize: batchsize to use for sequences, larger number goes faster, but needs more memory
    :type seq_batchsize: int
    :param seq_to_mutate: sequence to mutated and get embedding difference for each mutation
    :type seq_to_mutate: str
    :param bio_trans: biotransformer model
    :type bio_trans:
    :return:
    :rtype:
    """
    seqs_mutated = mutate_seq_insilico(seq_to_mutate)
    ref_embedding = bio_trans.compute_embeddings([seq_to_mutate], batch_size=embedding_batchsize)
    ref_embedding = ref_embedding['full'][0]

    seqs = list(seqs_mutated.keys())
    comb_batch = seq_batchsize
    n_batches = math.ceil(float(len(seqs)) / comb_batch)
    #print('total batches: ', str(n_batches))

    for batchi in range(n_batches):
        #print('Batch #', str(batchi))
        start = batchi * comb_batch
        end = (batchi + 1) * comb_batch
        subseqs = seqs[start:end]
        seqs_embeddings = bio_trans.compute_embeddings(subseqs, batch_size=embedding_batchsize)
        seqs_embeddings = seqs_embeddings['full']
        for i, embedding in enumerate(seqs_embeddings):
            seq = subseqs[i]
            sem_change = calc_array_distance(ref_embedding, embedding, l1_norm=l1_norm)
            meta = seqs_mutated[seq]
            meta['change'] = sem_change
        del seqs_embeddings

    mutations_change = {seqs_mutated[s]['mutation']: seqs_mutated[s]['change'] for s in seqs}
    return mutations_change


def calc_array_distance(array1, array2, l1_norm=True):
    """

    :param array1: reference (or sequence) array, can be embedding or attention matrix
    :type array1: np.ndarray
    :param array2: mutation sequence array, can be embedding or attention matrix
    :type array2: np.ndarray
    :param l1_norm: if true, change is based on l1_norm, does l2_norm otherwise. Default is true
    :type l1_norm: bool
    :return: matrix distance
    :rtype:
    """
    if l1_norm:
        change = abs(array1 - array2).sum()
    else:
        change = np.sqrt(np.sum((array1 - array2) ** 2))
    return change


def embedding_change_batchs(seqs, bio_trans, seq_batchsize, embedding_batchsize, seq_change, save_path=None,
                            chunksize=10, l1_norm=True):
    """
    takes a list of sequences, cuts them into chunks (size of chunksize),
    each chunk undergoes get_mutation_embedding_change and gets added to seq_change dict and saved

    :param l1_norm:
    :type l1_norm:
    :param save_path: if provided, save seq_change after calculating each chunk
    :type save_path: str
    :param seq_change: existing dict of key = sequence, value = mutations_change
    :type seq_change: dict
    :param seqs: list of sequences to get semantic change for
    :type seqs: list
    :param bio_trans: biotransformer model
    :type bio_trans:
    :param embedding_batchsize: batchsize to use in compute_embeddings
    :type embedding_batchsize: int
    :param seq_batchsize: batchsize to use for sequences, larger number goes faster, but needs more memory
    :type seq_batchsize: int
    :param chunksize: sized of chunked list, default is 10
    :type chunksize: int
    :return: seq_change
    :rtype: dict
    """
    seqs_chunked = list(chunks(seqs, chunksize))
    for seq_chunk in tqdm(seqs_chunked, desc='Chunked sequence list for semantic change'):
        for seq in tqdm(seq_chunk, desc='Sequences for semantic change'):
            mutations_change = get_mutation_embedding_change(seq, bio_trans, seq_batchsize, embedding_batchsize, l1_norm)
            seq_change[seq] = mutations_change
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(seq_change, f)
    return seq_change


def attention_change_batchs(seqs, bio_trans, tokenizer, device, seq_attn, save_path=None,
                            chunksize=10, pool_heads='max', pool_layer='max', l1_norm=False):
    seqs_chunked = list(chunks(seqs, chunksize))
    for seq_chunk in tqdm(seqs_chunked, desc='Chunked sequence list for attention change'):
        for seq in tqdm(seq_chunk, desc='Sequences for semantic change'):
            mutations_attn = get_mutation_attention_change(seq=seq, bio_trans=bio_trans, tokenizer=tokenizer,
                                                           device=device, pool_layer=pool_layer, pool_heads=pool_heads,
                                                           l1_norm=l1_norm)
            seq_attn[seq] = mutations_attn
            if save_path is not None:
                with open(save_path, 'wb') as f:
                    pickle.dump(seq_attn, f)
    return seq_attn


def get_mutation_attention_change(seq, bio_trans, tokenizer, device, pool_heads='max', pool_layer='max',
                                  l1_norm=False):
    ref_attn = get_attention(seq=seq, tokenizer=tokenizer, model=bio_trans, device=device, pool_heads=pool_heads)
    ref_attn = pool_attention(ref_attn, pool_layer=pool_layer)
    mutation_seqs = mutate_seq_insilico(seq)
    mutation_seqs = {mutation_seqs[s]['mutation']: s for s in list(mutation_seqs.keys())}
    mut_attn_changes = {}
    for mut, mut_seq in tqdm(mutation_seqs.items(), desc="Mutation Attention"):
        mut_attn = get_attention(mut_seq, tokenizer=tokenizer, model=bio_trans, device=device, pool_heads=pool_heads)
        mut_attn = pool_attention(mut_attn, pool_layer=pool_layer)
        attn_change = calc_array_distance(ref_attn, mut_attn, l1_norm=l1_norm)
        mut_attn_changes[mut] = attn_change
    return mut_attn_changes









