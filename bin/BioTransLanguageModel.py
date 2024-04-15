import math
import os
import pickle
from tqdm import tqdm
from DataHelpers import chunks, chunk_dictionary
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
    Calculates probabilities for a list of sequences, which have to be passed as a list of strings

    :param batchsize: batchsize for compute_probabilities
    :type batchsize: int
    :param bio_trans: biotransformer model
    :type bio_trans: biotransformers.wrappers.transformers_wrappers.TransformersWrapper
    :param seqs: list of sequences to compute probabiltiies for
    :type seqs: list
    :param forward_mode: if true, do forward mode for biotrans
    :type forward_mode: bool
    :return: dictionary of key = sequence, value = predicted probabilities
    :rtype: dict
    """
    if forward_mode:
        probabilities = bio_trans.compute_probabilities(seqs, pass_mode='forward', batch_size=batchsize)
    else:
        probabilities = bio_trans.compute_probabilities(seqs, pass_mode='masked', batch_size=batchsize)
    seq_probabilities = dict(zip(seqs, probabilities))
    return seq_probabilities


def compute_probabilities(bio_trans, seqs, batchsize, chunksize=None, forward_mode=False, save_path=None,
                          prior_seq_probabilities=None):
    """
    function to compute probabilities for a list of sequences
    Uses ability to chunk sequences to decrease working memory for biotransformer wrapper

    :param prior_seq_probabilities: starting dictionary of seq_probabilities
    :type prior_seq_probabilities: dict
    :param save_path: path to save chunks of seq_probabilities
    :type save_path:
    :param chunksize: if provided, cut the list of sequences to chunks (sized by chunksize)
    :type chunksize: int
    :param batchsize: batchsize for compute_probabilities
    :type batchsize: int
    :param bio_trans: biotransformer model
    :type bio_trans: biotransformers.wrappers.transformers_wrappers.TransformersWrapper
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
    Loads fine tuned biotransformer if given a model path to a checkpoint file
    Uses the biotransformers wrapper. This has some error when loading fine tuned model with multi-gpu, thus limits
    gpu to 1 if loading.

    :param model_path: path to biotransformer model checkpoint
    :type model_path: str
    :return: biotransformer wrapper
    :rtype: biotransformers.wrappers.transformers_wrappers.TransformersWrapper
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        # can be an issue with multi-gpu when loading bio_trans for inference
       n_gpu = 1
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    bio_trans = BioTransformers(backend="protbert", num_gpus=n_gpu)
    if model_path is not None:
        print("Loading Model from: {}".format(model_path))
        bio_trans.load_model(model_path)
    return bio_trans


def get_alphabet_dataloader(tokenizer, model_dir):
    """
    Define an alphabet mapping for common method between protbert
    Helper function taken from Biotransformers package to help when loading transformer for attention
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
    Loads the transformer outside of biotransformer wrapper for calculating attention
    Biotransformer wrapper package doesn't give ability to output attention, so this has to be loaded outside of
    wrapper package if calculating attention

    :param device: torch device for loading
    :type device: torch.device
    :param model_path: path to finetuned checkpoint
    :type model_path: str
    :return: tokenizer and transformer model (transformers.models.bert.modeling_bert.BertForMaskedLM)
    :rtype: tuple
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


def format_attention_torch(attention, layers=None, heads=None):
    """
    Formats attention output from transformer model

    :param attention: attention output from transformer model
    :type attention: torch.Tensor
    :param layers: if provided return attention for specific layers in list of layer indices
    :type layers: list
    :param heads: if provided return attention for specific heads in list of head indices
    :type heads: list
    :return: formatted attention in form of num_layers x num_heads x seq_len x seq_len
    :rtype: torch.Tensor
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


def get_layer_attentions_per_sample(layer_attentions):
    """
    reshapes attention output so that it's a list of input samples and each attention weight ouutput becomes
    a matrix of num_layers x num_heads x seq_len x seq_len
    :param layer_attentions: attention output after detaching from device and creating numpy array
    :type layer_attentions: numpy.ndarray
    :return: list of layer attentions per sample
    :rtype: list
    """
    attn_samples = []
    if len(layer_attentions.shape) > 4:
        attn = np.transpose(layer_attentions, (1, 0, 2, 3, 4))
        for i in range(0, attn.shape[0]):
            attn_samples.append(attn[i])
    else:
        attn_samples.append(layer_attentions)
    return attn_samples


def get_attention_multiple_seqs(seqs, tokenizer, model, device):
    input_seq_ids = prep_sequences_for_attn(seqs=seqs, tokenizer=tokenizer)
    with torch.no_grad():
        attention = model(input_seq_ids.to(device))[-1]
    # layer_attentions = format_attention_np(attention)
    layer_attentions = format_attention_torch(attention)
    layer_attentions = layer_attentions.cpu().detach().numpy()
    sample_attentions = get_layer_attentions_per_sample(layer_attentions)
    return sample_attentions


def prep_sequences_for_attn(seqs, tokenizer):
    """
    preps sequences for attention. Sequences get separated with space before passed to tokenizer

    :param seqs: list of sequences in string format
    :type seqs: any
    :param tokenizer: bert model tokenizer
    :type tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer
    :return: formatted input sequences
    :rtype: torch.Tensor
    """
    if isinstance(seqs, str):
        seq_list = [seqs]
    else:
        seq_list = seqs
    separated_sequences_list = [" ".join(seq) for seq in seq_list]
    encoded_inputs = tokenizer(separated_sequences_list, return_tensors="pt",
                               padding=True).to("cpu")
    input_seq_ids = encoded_inputs['input_ids']
    return input_seq_ids


def get_attention(seq, tokenizer, model, device):
    """
    Get attention output for 1 sequence and return numpy array.
    Usage is for the primary sequence of interest.

    :param seq: sequence input for model
    :type seq: str
    :param tokenizer: tokenizer output from load_biotrans_for_attn
    :type tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer
    :param model: bio_trans output from load_biotrans_for_attn
    :type model: transformers.models.bert.modeling_bert.BertForMaskedLM
    :param device: torch device
    :type device: torch.device
    :return: numpy array attention weights for the sequence of interest
    :rtype: numpy.ndarray
    """
    input_seq_ids = prep_sequences_for_attn(seqs=seq, tokenizer=tokenizer)
    with torch.no_grad():
        attention = model(input_seq_ids.to(device))[-1]
    layer_attentions = format_attention_torch(attention)
    layer_attentions = layer_attentions.cpu().detach().numpy()
    return layer_attentions


def pool_attention(layer_attentions, pool_heads_max=True, pool_layer_max=True):
    """
    pools attention input of size (30,16,N,N) -> (layers, heads, N, N) where N = length of sequence
    Creates Attention Matrix A of size (N,N)
    pools heads first (axis 1) and then layers
    default is max, does mean otherwise

    :param layer_attentions: attention matrix of size (30,16,N,N)
    :type layer_attentions: numpy.ndarray
    :param pool_heads_max: If true, do max pooling of attention heads. Do mean otherwise
    :type pool_heads_max: bool
    :param pool_layer_max: If true, do max pooling of attention layers. Do mean otherwise
    :type pool_layer_max: bool
    :return: pooled attention of size (N,N)
    :rtype: numpy.ndarray
    """
    if pool_heads_max:
        attn = layer_attentions.max(axis=1)
    else:
        attn = layer_attentions.mean(axis=1)
    if pool_layer_max:
        attn = attn.max(axis=0)
    else:
        attn = attn.mean(axis=0)
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


def get_mutation_embedding_change(seq_to_mutate, bio_trans, seq_batchsize, embedding_batchsize, l1_norm=False,
                                  subset_mutations=None):
    """
    Calculate semantic change of all mutations for a sequence of interest
    using subset_mutations decreases the amount of mutated sequences to calculate for and pass to the
    biotransformer model.

    :param subset_mutations: list of subset mutations to include, allows to only consider some mutations
    :type subset_mutations: list
    :param l1_norm: if true, change is based on l1_norm, does l2_norm otherwise. Default is true
    :type l1_norm: bool
    :param embedding_batchsize: batchsize to use in compute_embeddings
    :type embedding_batchsize: int
    :param seq_batchsize: batchsize to use for sequences, larger number goes faster, but needs more memory
    :type seq_batchsize: int
    :param seq_to_mutate: sequence to mutated and get embedding difference for each mutation
    :type seq_to_mutate: str
    :param bio_trans: biotransformer model
    :type bio_trans: biotransformers.wrappers.transformers_wrappers.TransformersWrapper
    :return: dictionary of key = mutation string and value = semantic change
    :rtype: dict
    """
    seqs_mutated = mutate_seq_insilico(seq_to_mutate, subset_mutations=subset_mutations)
    ref_embedding = bio_trans.compute_embeddings([seq_to_mutate], batch_size=embedding_batchsize)
    ref_embedding = ref_embedding['full'][0]

    seqs = list(seqs_mutated.keys())
    comb_batch = seq_batchsize
    n_batches = math.ceil(float(len(seqs)) / comb_batch)
    # print('total batches: ', str(n_batches))

    for batchi in range(n_batches):
        # print('Batch #', str(batchi))
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
    Calculates difference between two arrays

    :param array1: reference (or sequence) array, can be embedding or attention matrix
    :type array1: np.ndarray
    :param array2: mutation sequence array, can be embedding or attention matrix
    :type array2: np.ndarray
    :param l1_norm: if true, change is based on l1_norm, does l2_norm otherwise. Default is true
    :type l1_norm: bool
    :return: matrix distance
    :rtype: float
    """
    if l1_norm:
        change = abs(array1 - array2).sum()
    else:
        change = np.sqrt(np.sum((array1 - array2) ** 2))
    return change


def embedding_change_batchs(seqs, bio_trans, seq_batchsize, embedding_batchsize, seq_change, save_path=None,
                            chunksize=10, l1_norm=False, seq_subset_mut_dict=None):
    """
    takes a list of sequences, cuts them into chunks (size of chunksize),
    each chunk undergoes get_mutation_embedding_change and gets added to seq_change dict and saved

    :param seq_subset_mut_dict: dictionary of key = seq, value = list of subset mutations if subsetting
    :type seq_subset_mut_dict: dict
    :param l1_norm: if True, use L1 norm difference, L2 otherwise
    :type l1_norm: bool
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
            if seq_subset_mut_dict is not None:
                subset_muts = seq_subset_mut_dict[seq]
            else:
                subset_muts = None
            mutations_change = get_mutation_embedding_change(seq_to_mutate=seq, bio_trans=bio_trans,
                                                             seq_batchsize=seq_batchsize,
                                                             embedding_batchsize=embedding_batchsize,
                                                             l1_norm=l1_norm, subset_mutations=subset_muts)
            seq_change[seq] = mutations_change
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(seq_change, f)
    return seq_change


def attention_change_batchs(seqs, model_path, seq_attn=None, save_path=None,
                            pool_heads_max=True, pool_layer_max=True, l1_norm=False, subset_mutations=None):
    """
    Calculate attention change for sequences in batches
    Incrementally saves output to save_path

    :param subset_mutations: needs to be a list of lists of subset_mutations for each seq in seqs, if using a subset
    :type subset_mutations: list
    :param model_path: path of finetuned checkpoint to load biotrans
    :type model_path: str
    :param seqs: list of seqs to mutate and get attention changes
    :type seqs: list
    :param seq_attn: prior dictionary of key=seq, value = mutation attention changes
    :type seq_attn: dict
    :param save_path: save path for seq_attn
    :type save_path: str
    :param pool_heads_max: if true, pool heads with max pooling
    :type pool_heads_max: bool
    :param pool_layer_max: if true, pool layers with max pooling
    :type pool_layer_max: bool
    :param l1_norm: to use L1 norm distance,default is False
    :type l1_norm: bool
    :return: dict where key = sequence, value = attention change for mutations
    :rtype: dict
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, bio_trans = load_biotrans_for_attn(device=device, model_path=model_path)
    if seq_attn is None:
        seq_attn_dict = {}
    else:
        seq_attn_dict = seq_attn
    for i, seq in enumerate(tqdm(seqs, desc='Sequences for attention change')):
        if subset_mutations is not None:
            subset_muts = subset_mutations[i]
        else:
            subset_muts = None
        mutations_attn = get_mutation_attention_change(seq_to_mutate=seq, bio_trans=bio_trans, tokenizer=tokenizer,
                                                       device=device, pool_heads_max=pool_heads_max,
                                                       pool_layer_max=pool_layer_max, l1_norm=l1_norm,
                                                       subset_mutations=subset_muts)
        seq_attn_dict[seq] = mutations_attn
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(seq_attn_dict, f)
    return seq_attn_dict


def get_mutation_attention_change(seq_to_mutate, bio_trans, tokenizer, device, pool_heads_max=True,
                                  pool_layer_max=True, l1_norm=False, subset_mutations=None):
    """
    uses multi processing with Ray to calculate attention change for a single sequence
    attention change is calculated from passing each mutated sequence into the bio transformer and returning attention
    matrices and attention change is calculated as difference betwen sequence of interest and the mutated sequence
    Can be computationally expensive so uses multi-processing with Ray for efficiency

    :param seq_to_mutate: sequence of interest
    :type seq_to_mutate: str
    :param bio_trans: bio_trans output from load_biotrans_for_attn
    :type bio_trans: transformers.models.bert.modeling_bert.BertForMaskedLM
    :param tokenizer: tokenizer output from load_biotrans_for_attn
    :type tokenizer: transformers.models.bert.tokenization_bert.BertTokenizer
    :param device: torch device
    :type device: torch.device
    :param pool_heads_max: if true pool heads with max, mean otherwise. Default True
    :type pool_heads_max: bool
    :param pool_layer_max: if true pool layers with max, mean otherwise. Default True
    :type pool_layer_max: bool
    :param l1_norm: if true, use l1 norm, use l2 norm otherwise. default is false
    :type l1_norm: bool
    :param subset_mutations: if provided, calc attention change for only this subset of mutations
    :type subset_mutations: list
    :return: dictionary of key = mutation, value = attention change
    :rtype: dict
    """
    if ray.is_initialized():
        ray.shutdown()
        torch.cuda.empty_cache()
    ray.init()
    ref_attn = get_attention(seq=seq_to_mutate, tokenizer=tokenizer, model=bio_trans, device=device)
    ref_attn = pool_attention(ref_attn, pool_heads_max=pool_heads_max, pool_layer_max=pool_layer_max)
    mutation_seqs = mutate_seq_insilico(seq_to_mutate, subset_mutations=subset_mutations)
    mutation_seqs = {mutation_seqs[s]['mutation']: s for s in list(mutation_seqs.keys())}

    # Set up ray actors
    model_ref = ray.put(bio_trans)
    n_gpu = torch.cuda.device_count()
    actor_pool = [BertActor.remote(model=model_ref, device=device, tokenizer=tokenizer) for _ in range(n_gpu)]
    # Parallelize the predict_attention() method over the list of mutation dictionaries
    results = []
    for mut, mut_seq in mutation_seqs.items():
        actor = actor_pool.pop(0)  # Take an actor from the pool
        results.append(actor.predict_attention.remote(mut=mut, seq=mut_seq, pool_heads_max=pool_heads_max,
                                                      pool_layer_max=pool_layer_max))
        actor_pool.append(actor)  # Return the actor to the pool

    # Collect the results
    result_dict = {}
    pbar = tqdm(total=len(results))
    while len(results) > 0:
        done_ids, results = ray.wait(results)
        for done_id in done_ids:
            result_dict = process_incremental(results=result_dict, mut_attn_dict=ray.get(done_id), ref_attn=ref_attn,
                                              l1_norm=l1_norm)
            pbar.update(1)
    pbar.close()
    ray.shutdown()
    return result_dict


# Define a Ray actor class that wraps the model
@ray.remote(num_cpus=5, num_gpus=1)
class BertActor:
    def __init__(self, model, device, tokenizer):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

    def predict_attention(self, mut, seq, pool_heads_max=True, pool_layer_max=True):
        layer_attentions = get_attention(seq=seq, tokenizer=self.tokenizer, model=self.model, device=self.device)
        pooled_attn = pool_attention(layer_attentions, pool_heads_max=pool_heads_max, pool_layer_max=pool_layer_max)
        return {mut: pooled_attn}


def process_incremental(results, mut_attn_dict, ref_attn, l1_norm):
    """
    calc function that drives attention change calculation during multiprocessing for attention change

    :param l1_norm: if true, use l1 norm, use l2 norm otherwise
    :type l1_norm: bool
    :param results: mutation attention change results to add to
    :type results: dict
    :param mut_attn_dict: dict of key = mutation, value = attention matrix
    :type mut_attn_dict: dict
    :param ref_attn: sequence of interest attention matrix
    :type ref_attn: numpy.ndarray
    :return: adds to results dict
    :rtype: dict
    """
    for mut, mut_attn in mut_attn_dict.items():
        attn_change = calc_array_distance(array1=ref_attn, array2=mut_attn, l1_norm=l1_norm)
        results.update({mut: attn_change})
    return results


def process_incremental_mean_attn(results, attn_dict, index_seq_dict):
    """
    Function used when calculating mean attention matrices for sequences in tree for visualization

    :param index_seq_dict: dict of key =index, value = sequence
    :type index_seq_dict: dict
    :param results: reuslts dict to add to
    :type results: dict
    :param attn_dict: attention  matrix dict
    :type attn_dict: dict
    :return: added results dict
    :rtype: dict
    """
    for index, attn in attn_dict.items():
        seq = index_seq_dict[index]
        mean_attn = attn.mean(axis=0)
        results[seq] = mean_attn
    return results
