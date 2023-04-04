"""
module holds some functions for MSA, entropy calculation, PSSM calculations, etc.
"""
import pandas as pd
from Bio import AlignIO, SeqIO
from tqdm import tqdm
import numpy as np
from MutationHelpers import check_known_aa
import math
import pickle
from Bio.Align import AlignInfo
from MutationHelpers import load_ref_spike


def filter_nextclade_aligned(nextclade_folder="data/interim/nextclade_results"):
    """
    function to take the alignemnt from nextclade, does subsequent filtering and saving the new fasta
    filter for accessions that passed nextclade QC
    filter for unique sequences
    filter for sequences with non-ambiguous (known) AA
    :param nextclade_folder: folder to find nextclade alignment and to save filtered fastas
    :type nextclade_folder: str
    :return: nothing, just saves fastas at different steps
    """
    filename = nextclade_folder + "/nextclade_gene_S.translation.fasta"
    alignment = AlignIO.read(filename, "fasta")
    df = pd.read_pickle('data/MetaData/sequences_metadata_final.pkl')
    accessions = df['Nucleotide Accession'].values.tolist()
    alignment_len = alignment.get_alignment_length()
    n_seq = len(alignment)
    msg = "Alignment Length: {} \n N Sequences in Alignment: {} \n".format(alignment_len, n_seq)
    pass_qc_aln = []
    for aln in tqdm(alignment, desc='Full Alignment'):
        if aln.id in accessions:
            pass_qc_aln.append(aln)
    msg = msg + "N Seq passed QC: {} \n".format(len(pass_qc_aln))
    save_path = nextclade_folder + "/nextclade.gene.S.filtered.fasta"
    SeqIO.write(pass_qc_aln, save_path, "fasta")
    unique_aln = {}
    for aln in tqdm(pass_qc_aln, desc='Filtered Alignment'):
        if str(aln.seq) not in unique_aln:
            unique_aln[aln.seq] = aln
    msg = msg + "N Unique Seq passed QC: {} \n".format(len(unique_aln))
    unique_aln_seq = list(unique_aln.values())
    save_path = nextclade_folder + "/nextclade.gene.S.filtered_unique.fasta"
    SeqIO.write(unique_aln_seq, save_path, "fasta")
    knownaa_aln = []
    for aln in tqdm(unique_aln_seq, desc='Unique Filtered Alignment'):
        if check_known_aa(str(aln.seq), extra_tokens=['*', '-']):
            knownaa_aln.append(aln)
    msg = msg + "N Non-Ambiguous Unique Seq Passed QC: {}".format(len(knownaa_aln))
    save_path = nextclade_folder + "/nextclade.gene.S.filtered_unique_knownaa.fasta"
    SeqIO.write(knownaa_aln, save_path, "fasta")
    print(msg)
    with open(nextclade_folder + '/nextclade_alignment_info.txt', 'w') as f:
        f.write(msg)


def parseMSA(msa_filepath, alnformat='FASTA'):
    """
    Parse in the MSA file using Biopython's AlignIO

    :param msa_filepath:
    :type msa_filepath:
    :param alnformat:
    :type alnformat:
    :return:
    :rtype:
    """
    alignment = AlignIO.read(msa_filepath, alnformat)

    seq_lengths_list = []
    for record in alignment:
        seq_lengths_list.append(len(record))

    seq_lengths = set(seq_lengths_list)

    print("Alignment length is:" + str(list(seq_lengths)))

    if len(seq_lengths) != 1:
        print("Your alignment lengths aren't equal. Check your alignment file.")
        return np.nan

    index = range(0, list(seq_lengths)[0])
    return alignment, list(seq_lengths), index


def shannon_entropy(list_input):
    """
    Calculate Shannon's Entropy per column of the alignment (H=-\sum_{i=1}^{M} P_i\,log_2\,P_i)
    https://www.biostars.org/p/258014/

    :param list_input: column of alignment
    :type list_input: list
    :return: entropy of column position
    :rtype: float
    """

    unique_base = set(list_input)
    M = len(list_input)
    entropy_list = []
    # Number of residues in column
    for base in unique_base:
        n_i = list_input.count(base)  # Number of residues of type i
        P_i = n_i / float(M)  # n_i(Number of residues of type i) / M(Number of residues in column)
        entropy_i = P_i * (math.log(P_i, 2))
        entropy_list.append(entropy_i)

    sh_entropy = -(sum(entropy_list))
    return sh_entropy


def shannon_entropy_list_msa(alignment):
    """
    Calculate Shannon Entropy across the whole MSA
    https://www.biostars.org/p/258014/
    :param alignment: alignment to calculate entropy
    :type alignment: Bio.Align.MultipleSeqAlignment
    :return: entropy for each position
    :rtype: list
    """

    shannon_entropy_list = []
    for col_no in range(0, len(list(alignment[0]))):
        list_input = list(alignment[:, col_no])
        shannon_entropy_list.append(shannon_entropy(list_input))

    return shannon_entropy_list


def position_entropy_dict(alignment_filename, save_path=None):
    """
    calculates shannon entropy and creates dictionary of key = position, value = entropy
    :param save_path: path to save entropy dict
    :type save_path: str
    :param alignment_filename: path of alignment
    :type alignment_filename: str
    :return:
    :rtype:
    """
    alignment, seq_lengths, index = parseMSA(alignment_filename)
    entropy = shannon_entropy_list_msa(alignment)
    entropy_dict = dict(zip(index, entropy))
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(entropy_dict, f)


def get_pssm(alignment, ignore_invalid=True):
    """
    get position scoring matrix from alignment
    if ignore_invalid, only base 20 AA are considered

    :param ignore_invalid: if ignore_invalid, only base 20 AA are considered
    :type ignore_invalid: bool
    :param alignment: alignment to score
    :type alignment: Bio.Align.MultipleSeqAlignment
    :return: pssm
    :rtype: Bio.Align.AlignInfo.PSSM
    """
    if ignore_invalid:
        ignore_chars = ['X', 'U', 'O', 'B', 'Z', 'J', 'X', '*']
    else:
        ignore_chars = []
    summary_align = AlignInfo.SummaryInfo(alignment)
    pssm = summary_align.pos_specific_score_matrix(chars_to_ignore=ignore_chars)
    return pssm


def entropy_calc_pssm(pssm, ref_record=None):
    """
    calculates entropy for each position in PSSM

    :param ref_record: reference record that MSA was aligned against
    :type ref_record:
    :param pssm:
    :type pssm: Bio.Align.AlignInfo.PSSM
    :return: dictionary of key = position index, value = entropy of position from PSSM
    :rtype: dict
    """
    if ref_record is None:
        ref = load_ref_spike()
        ref = str(ref.seq)
    else:
        ref = str(ref_record.seq)
    entropy_dict = {}
    for i in range(0, len(ref)):
        ent = entropy_pos(pssm[i])
        entropy_dict[i] = ent
    return entropy_dict


def entropy_pos(pssm_dict):
    """
    calculates entropy from a given position dictionary from PSSM
    :param pssm_dict: dictionary of key=position, values = amino acids and count from MSA
    :type pssm_dict: dict
    :return: shannon entropy
    :rtype: float
    """
    M = sum(pssm_dict.values())
    unique_base = []
    for k, v in pssm_dict.items():
        if v > 0:
            unique_base.append(k)
    entropy_list = []
    # Number of residues in column
    for base in unique_base:
        n_i = pssm_dict[base]  # Number of residues of type i
        P_i = n_i / float(M)  # n_i(Number of residues of type i) / M(Number of residues in column)
        entropy_i = P_i * (math.log(P_i, 2))
        entropy_list.append(entropy_i)
    return -(sum(entropy_list))


def calc_position_muts(pssm, ref_record=None):
    """
    over a given PSSM, summate the number of sequences from MSA that don't have the same AA in same position as ref
    :param pssm: position scoring matrix from MSA
    :type pssm: Bio.Align.AlignInfo.PSSM
    :param ref_record: Reference spike record
    :type ref_record: Bio.SeqRecord.SeqRecord
    :return: dictionary of key = position, value = # sequences with mutations
    :rtype: dict
    """
    if ref_record is None:
        ref = load_ref_spike()
        ref = str(ref.seq)
    else:
        ref = str(ref_record.seq)
    pos_mut_dict = {}
    for i in range(0, len(ref)):
        ref_aa = ref[i]
        n_muts = 0
        for k, v in pssm[i].items():
            if k != ref_aa:
                n_muts = n_muts + v
        pos_mut_dict[i] = n_muts
    return pos_mut_dict


def cutoff_alignment(cut_off='2022-1-1'):
    """
    takes the filtered_unique alignment and separates into pre and post cut off alignments
    :param cut_off: date for cutoff
    :type cut_off: str
    :return: saves pre and post cutoffs
    :rtype:
    """
    df = pd.read_pickle('data/MetaData/sequences_metadata_final.pkl')
    pre_cutoff = df[(df['Collection Date'] < cut_off)]['Nucleotide Accession'].values.tolist()
    print('Number Accessions in pre cutoff: {}'.format(len(pre_cutoff)))
    post_cutoff = df[(df['Collection Date'] >= cut_off)]['Nucleotide Accession'].values.tolist()
    print('Number Accessions in post cutoff: {}'.format(len(post_cutoff)))
    del df
    filename = 'data/interim/nextclade_results/nextclade.gene.S.filtered_unique.fasta'
    alignment, seq_lengths, index = parseMSA(filename, "fasta")
    post_aln = []
    pre_aln = []
    for aln in tqdm(alignment, desc='Filtered Unique Alignment'):
        if aln.id in post_cutoff:
            post_aln.append(aln)
        elif aln.id in pre_cutoff:
            pre_aln.append(aln)
    print('Number Accessions in pre cutoff Alignment: {}'.format(len(pre_cutoff)))
    print('Number Accessions in post cutoff Alignment: {}'.format(len(post_cutoff)))
    nextclade_folder = "data/interim/nextclade_results"
    save_path = nextclade_folder + "/nextclade.gene.S.filtered_unique_postcutoff.fasta"
    SeqIO.write(post_aln, save_path, "fasta")

    save_path = nextclade_folder + "/nextclade.gene.S.filtered_unique_precutoff.fasta"
    SeqIO.write(pre_aln, save_path, "fasta")
    msg = 'N Filtered Unique PreCutoff:{} \n N Filtered Unique Postcutoff:{}'.format(len(pre_aln), len(post_aln))
    text_path = 'data/interim/nextclade_results/nextclade_alignment_info.txt'
    with open(text_path, 'a+') as f:
        f.write(msg)


def aln_to_pos_muts(aln_path, save_name):
    """
    takes an alignment file, creates PSSM, and runs calc_position_muts
    saves in data/processed/position_stats
    :param save_name: name for file (specify if pre or post cut)
    :type save_name: str
    :param aln_path: path to alignment fasta
    :type aln_path: str
    :return: just saves dict of key = pos, value = n muts
    :rtype:
    """
    alignment, seq_lengths, index = parseMSA(aln_path, "fasta")
    print('N seq in alignment: {}'.format(len(alignment)))
    pssm = get_pssm(alignment)
    pos_mut_dict = calc_position_muts(pssm)
    folder = 'data/processed/position_stats//'
    print('Saving')
    with open(folder+save_name+'.pkl', 'wb') as f:
        pickle.dump(pos_mut_dict, f)
    print('Done')


if __name__ == '__main__':

    filter_nextclade_aligned()
    print('doing cutoff')
    cutoff_alignment()

    print('Precut off position muts')
    precut_path = "data/interim/nextclade_results/nextclade.gene.S.filtered_unique_precutoff.fasta"
    aln_to_pos_muts(precut_path, 'pos_muts_pssm_precutoff')
    print('Postcut off position muts')
    postcut_path = "data/interim/nextclade_results/nextclade.gene.S.filtered_unique_postcutoff.fasta"
    aln_to_pos_muts(postcut_path, 'pos_muts_pssm_postcutoff')
    
    full_path = 'data/interim/nextclade_results/nextclade.gene.S.filtered_unique.fasta'
    aln_to_pos_muts(full_path, 'pos_muts_pssm_all')





