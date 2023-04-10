"""
functions to compare, translate, align, mutate, sequences
some helper functions to perform calculations based on AA posterior probabilities
Helper functions to convert probabilities dictionary into forms needed for other modules
Helper functions to prep position conservation datasets
"""
import pathogenie
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import pandas as pd


def load_ref_spike(ref_path="data/ref_fastas/reference_spike.fasta"):
    """
    returns the reference sequence record
    :param ref_path: path to reference spike record fasta
    :type ref_path: str
    :return: reference spike record
    :rtype: Bio.SeqRecord.SeqRecord
    """
    ref_spike_record = list(SeqIO.parse(ref_path, 'fasta'))[0]
    return ref_spike_record


def remove_end_star(seq):
    """
    helper function
    some sequences end with "*" symbol and this will remove it
    :param seq: genomic sequence
    :type seq: str
    :return: sequence without * at end
    :rtype: str
    """
    if seq[-1] == '*':
        seq = seq[:-1]
    return seq


def compare_two_seq(s1, s2):
    """
    function to help view differences between two sequences
    :param s1: first sequence
    :type s1: str
    :param s2: second sequence
    :type s2: str
    :return: prints the differences between them
    :rtype: None
    """
    seq1 = remove_end_star(s1)
    seq2 = remove_end_star(s2)
    for i, aa in enumerate(seq1):
        # if seq1 is longer than seq2
        if i >= len(seq2):
            print('Sequence 1: ' + str(i) + seq1[i])
            print('Sequence 2: End')
        else:
            if seq1[i] != seq2[i]:
                print('Sequence 1: ' + str(i) + seq1[i])
                print('Sequence 2: ' + str(i) + seq2[i])
    # if seq2 is longer than seq1
    while i < len(seq2) - 1:
        print('Sequence 1: End')
        print('Sequence 2: ' + str(i) + seq2[i])
        i = i + 1
    print('done comparing')


def mutate_sequence(seq, mutations, ignore_deletion=True, early_stop=True):
    """
    function to change a sequence based on list of mutations
    mutations are such as reference AA [position from 1] mutated AA
    deletion tokens are "-" if ignore_deletion, then remove those from the string
    :param seq: sequence to mutate
    :type seq: str
    :param mutations: list of mutations
    :type mutations: list
    :param ignore_deletion: if true, remove deletion tokens
    :type ignore_deletion: bool
    :param early_stop: if true and "*" mutations are listed, stop the sequence after the stop codon
    :type early_stop: bool
    :return: mutated sequence
    :rtype: str
    """
    mutated_seq = seq
    for mut in mutations:
        mutable = mut[-1]
        ref = mut[0]
        pos = int(mut[1:-1]) - 1
        # check against the current mutated_seq incase of multiple mutations
        if mutated_seq[pos] != ref:
            print("Mutation listed isn't found in sequence")
            print(mut)
            return
        mutated_seq = mutated_seq[:pos] + mutable + mutated_seq[pos + 1:]

    if ignore_deletion:
        mutated_seq = mutated_seq.replace('-', '')
    if early_stop:
        codon_index = mutated_seq.find('*')
        if codon_index != -1:
            mutated_seq = mutated_seq[0:codon_index]
    return mutated_seq


def sort_mut_list(muts):
    """
    function to take a list of mutations and sort them by their position
    only works if positions are unique
    :param muts: list of mutations
    :type muts: lst
    :return: sorted lst by position
    :rtype: lst
    """
    mut_dict = {}
    # muts = node_muts_from_ref(muts)
    for mut in muts:
        pos = int(mut[1:-1])
        mut_dict[pos] = mut
    sorted_keys = sorted(mut_dict)
    sorted_muts = [mut_dict[k] for k in sorted_keys]
    return sorted_muts


def mutate_seq_insilico(seq, amino_acids=None, significant_mutations=None):
    """

    :param seq:
    :type seq:
    :param amino_acids:
    :type amino_acids:
    :param significant_mutations:
    :type significant_mutations:
    :return:
    :rtype:
    """
    if amino_acids is None:
        AAs = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
            'Y', 'V',
        ]
    else:
        AAs = amino_acids
    seqs_mutated = {}
    # for each position of the reference sequence
    for idx in range(len(seq)):
        # for each amino acid in list (of 20)
        for aa in AAs:
            # if they equal then skip
            if aa == seq[idx]:
                continue
            # otherwise swap them out, to make the mutated sequence
            mut_seq = seq[:idx] + aa + seq[idx + 1:]
            # mutated string summarizes the mutation that was created
            mut_str = '{}{}{}'.format(seq[idx], idx + 1, aa)
            if mut_seq not in seqs_mutated:
                meta = {
                    'mutation': mut_str
                }
                if significant_mutations is not None:
                    meta['significant'] = mut_str in significant_mutations
                seqs_mutated[mut_seq] = meta
    return seqs_mutated


def get_mutation_probabilities(seq, probabilities, seqs_mutated_dict, amino_acids=None, prob_col='prob'):
    if amino_acids is None:
        AAs = [
            'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
            'Y', 'V',
        ]
    else:
        AAs = amino_acids
    min_pos = 0
    max_pos = len(seq) - 1
    seqs_mutated = seqs_mutated_dict
    word_pos_prob = {}
    for i in range(min_pos, max_pos + 1):
        for aa in AAs:
            if seq[i] == aa:
                continue
            pos_dict = probabilities[i]
            prob = pos_dict[aa]
            word_pos_prob[(aa, i)] = prob
    for (word, pos), prob in word_pos_prob.items():
        mutable = seq[:pos] + word + seq[pos + 1:]
        meta = seqs_mutated[mutable]
        meta['mut'] = word
        meta['pos'] = pos
        meta[prob_col] = prob
    return seqs_mutated


def aln_parent_child_node(parent_node, child_node):
    """
    takes 2 nodes from PhyloTreeParsers, creates the seqrecord for them and runs get_aln_muts

    :param parent_node: parent node
    :type parent_node:  PhyloTreeParsers.Node
    :param child_node: child node
    :type child_node: PhyloTreeParsers.Node
    :return: aligned muts
    :rtype: list
    """
    parent_record = build_seqrecord(parent_node.spike_seq, parent_node.name)
    child_record = build_seqrecord(child_node.spike_seq, child_node.name)
    aln_muts = get_aln_muts(parent_record, child_record)
    return aln_muts


def get_aln_muts(reference, sequence):
    """
    aligns 2 sequences
    reference can be any sequence, not necessarily reference sequence
    could be parent and child sequences

    :param reference: reference sequence is based compared
    :type reference:
    :param sequence: sequence to align to
    :type sequence:
    :return: list of aligned mutations
    :rtype: list
    """
    if isinstance(reference, str):
        ref_record = build_seqrecord(reference, "ref_record")
    else:
        ref_record = reference
    if isinstance(sequence, str):
        seq_record = build_seqrecord(sequence, 'interest_record')
    else:
        seq_record = sequence
    aln = pathogenie.clustal_alignment(seqs=[ref_record, seq_record])
    aln_muts = []
    for pos in range(len(aln[0])):
        refaa = aln[0, pos]
        aa = aln[1, pos]
        if aa != refaa:
            mut = refaa + str(pos + 1) + aa
            aln_muts.append(mut)
    return aln_muts


def build_seqrecord(sequence, sequence_name=None):
    """
    builds SeqRecord from string sequence and sequence name, if name isn't provided, build record with generic name
    not sure if needed, but adds "*" at end for pathogenie

    :param sequence_name: name for seqrecord
    :type sequence_name: str
    :param sequence: string sequence
    :type sequence: str
    :return: Seq Record
    :rtype: Bio.SeqRecord.SeqRecord
    """
    seq = sequence + "*"
    if sequence_name is not None:
        name = sequence_name
    else:
        name = "Sequence Record"
    seq_record = SeqRecord(Seq(seq), id=name)
    return seq_record


def get_seq_mutation_dict(seq, probabilities, changes=None, significant_mutations=None):
    """
    make a dictionary of key = mutation, value = meta (with mutation, wt, pos, mut, significant, probability and change)
    :param seq: sequence to find mutations from
    :type seq: str
    :param probabilities: dictionary of probabilities from bio_trans
    :type probabilities: dict
    :param changes: dictionary of key = mutation str, value = semantic change
    :type changes: dict
    :param significant_mutations: if present, add significant to meta
    :type significant_mutations: list
    :return: dictionary of key = mutation, value = meta values
    :rtype: dict
    """
    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V',
    ]
    seq_mutations = {}
    for idx in range(len(seq)):
        for aa in AAs:
            if aa == seq[idx]:
                continue
            mut_str = '{}{}{}'.format(seq[idx], idx + 1, aa)
            meta = {
                'mutation': mut_str,
                'wt': seq[idx],
                'pos': idx,
                'mut': aa,
            }
            if significant_mutations is not None:
                meta['significant'] = mut_str in significant_mutations
            seq_mutations[mut_str] = meta
    for pos, pos_proba in probabilities.items():
        for aa, prob in pos_proba.items():
            if seq[pos] == aa:
                continue
            mutation = '{}{}{}'.format(seq[pos], pos + 1, aa)
            meta = seq_mutations[mutation]
            meta['prob'] = prob
            if changes is not None:
                meta['change'] = changes[mutation]
    return seq_mutations


def translate_seq(dna_seq, range_list):
    """
    function to take a DNA sequence and range_list of [start,stop] positions in dna sequence
    sequence is then translated
    :param dna_seq: dna sequence
    :type dna_seq: Bio.Seq.Seq
    :param range_list: list of [start, stop] dna positions
    :type range_list: list
    :return: translated sequence, if sequence doesn't have triplet codons, returns null
    :rtype: str
    """
    start = range_list[0] - 1
    end = range_list[1]
    seq_to_translate = dna_seq[start:end]
    # check if we have triplet codons
    if not len(seq_to_translate) % 3 == 0:
        return np.NaN
    translated = seq_to_translate.translate()
    translated = str(translated)[0:-1]
    return translated


def check_known_aa(sequence, extra_tokens=None):
    """
    takes a protein sequence and if it only contains the 20 base AA then return true
    can add additional tokens to consider, such as '-" and "*"
    :param sequence: sequence to check
    :type sequence: str
    :param extra_tokens: list of extra tokens to add
    :type extra_tokens: list
    :return: True if sequence only has base AAs, False otherwise
    :rtype: bool
    """
    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V',
    ]
    if extra_tokens is not None:
        AAs.extend(extra_tokens)
    AA_SET = set(('').join(AAs))
    return not any(x not in AA_SET for x in sequence)


def marginal_score(mutation, probabilities):
    """
    mutation marginal score of a given mutation and probabilities dictionary
    :param mutation: mutant string formatted as  [WT_AA][POS][MUT_AA], with pos starting from 1
    :type mutation: str
    :param probabilities: dictionary of predicted probabilities from bio_trans, key = position, values = proba dict
    :type probabilities: dict
    :return: marginal score
    :rtype: float
    """
    wt = mutation[0]
    pos = int(mutation[1:-1]) - 1
    mut = mutation[-1]
    score = np.log(probabilities[pos][mut]) - np.log(probabilities[pos][wt])
    return score


def log_likelihood(seq, probabilites):
    """
    calculates log likelihood of each amino acid from probabiltiies dictioanry
    :param seq: sequence to calculate log likelihood for
    :type seq: str
    :param probabilites: dictionary of predicted probabilities from bio_trans, key = position, values = proba dict
    :type probabilites: dict
    :return: log likelihood of sequence
    :rtype: float
    """
    log = np.sum([np.log(probabilites[i][seq[i]]) for i in range(len(seq))])
    log = log / len(seq)
    return log


def ref_seq_pos_map(reference, sequence=None, mutations=None):
    """
    since a given sequence might have deletions to shift positions relative to reference, this will map position indices
    from reference to sequence

    :param mutations: list of mutations to create the position mapping
    :type mutations: list
    :param reference: reference sequence
    :type reference: str
    :param sequence: sequence of interest
    :type sequence: str
    :return: dictionary of key = reference positon, value = sequence position, deletions are "-"
    :rtype: dict
    """
    if mutations is None and sequence is None:
        raise ValueError('Need at least a list of mutations or a sequence to create a list of mutations')
    if mutations is not None:
        aln_muts = sort_mut_list(node_muts_from_ref(mutations))
    else:
        aln_muts = get_aln_muts(reference, sequence)
        aln_muts = sort_mut_list(aln_muts)
    dels = []
    for mut in aln_muts:
        pos = int(mut[1:-1])
        alt = mut[-1]
        if alt == '-':
            # have to append pos - 1 b/c the mutations have indexed from 1 and positions are indexed from 0
            dels.append(pos - 1)
    pos_map = {}
    i = 0
    j = 0
    while i < len(reference):
        if i in dels:
            pos_map[i] = '-'
        else:
            pos_map[i] = j
            j = j + 1
        i = i + 1
    return pos_map


def remap_to_seq(position_map, position_dict):
    """
    takes dictionaries that have keys = position (in reference), and a position mapping dict (from output of
    ref_seq_pos_map) and returns the values remapped to sequence position

    :param position_map: dict of key=ref positon, value = sequence position
    :type position_map: dict
    :param position_dict: dict of key=ref position, value = any position related value
    :type position_dict: dict
    :return: dict of key=sequence position, value = position related value
    :rtype: dict
    """
    dict_to_seq = {}
    for ref_pos, alt_pos in position_map.items():
        if alt_pos == '-':
            continue
        dict_to_seq[alt_pos] = position_dict[ref_pos]
    return dict_to_seq


def proba_to_df(seq, probabilities, sig_muts=None):
    """
    takes the probabilities from biotransformer and creates a dataframe
    marks which AA is WT (current AA in sequence)
    unlike other functions that transforms the biotransformer probabilities, this includes WT AA
    :param sig_muts: if list is provided, then mark true/false if mutation is significant
    :type sig_muts: list
    :param seq: sequence of interest
    :type seq: str
    :param probabilities: dict of probabilities from biotransformer
    :type probabilities: dict
    :return: dataframe of values
    :rtype: pandas.DataFrame
    """
    data = []

    for i in range(0, len(probabilities)):
        for aa, prob in probabilities[i].items():
            if seq[i] == aa:
                mut = 'WT'
            else:
                mut = seq[i] + str(i + 1) + aa
            data_dict = {
                'mut': aa,
                'pos': i,
                'mutation': mut,
                'prob': prob,
            }
            if sig_muts is not None:
                data_dict['sig_mut'] = mut in sig_muts
            data.append(data_dict)
    df = pd.DataFrame.from_records(data)
    return df


def pullout_pos_mut(mutation):
    """
    takes a string like "D614G" and returns the WT (D), position (613) and Alt (G)
    :param mutation: mutation string
    :type mutation: str
    :return: wt, pos, alt (mut)
    :rtype: tuple
    """
    mut_type = label_mut_type(mutation)
    if mut_type == 'ins':
        i = mutation.find(':')
        wt = np.nan
        pos = int(mutation[:i]) - 1
        mut = mutation[i + 1:]
    else:
        wt = mutation[0]
        pos = mutation[1:-1]
        pos = int(pos) - 1
        mut = mutation[-1]
    return wt, pos, mut


def pullout_pos_mut_df(row, col='mutation'):
    """
    formatted to call pullout_pos_mut on a dataframe that has the mutation string a column under "Mutation"
    :param row:
    :type row:
    :return:
    :rtype:
    """
    mutation = row[col]
    wt, pos, mut = pullout_pos_mut(mutation)
    return wt, pos, mut


def collapse_muts(mutations):
    """
    takes a list of mutations, if the position index is the same, it collapses into 1 mutation
    i.e. D614F and F614G get collapsed to D614G

    :param mutations: list of mutations
    :type mutations: list
    :return: collapsed list of mutations
    :rtype: list
    """
    i = 0
    j = i + 1
    current = mutations[i]
    while j < len(mutations):

        next_mut = mutations[j]
        if current[-1] == next_mut[0]:
            current = current[0:-1] + next_mut[-1]
        i = i + 1
        j = j + 1
    if current[0] == current[-1]:
        return []
    else:
        return [current]


def node_muts_from_ref(mutations):
    """
    takes a list of mutations from a node (node.spike_mutations) and collapses them such that the muations are
    with reference to the reference sequence (WT is the reference)
    :param mutations: list of mutations
    :type mutations: list
    :return: list of mutations with WT as reference sequence
    :rtype: list
    """
    mutted = {}
    for mut in mutations:
        pos = mut[1:-1]
        if pos not in mutted:
            mutted[pos] = []
        mutted[pos].append(mut)
    final_muts = []
    for pos, muts in mutted.items():
        if len(muts) == 1:
            final_muts.extend(muts)
        else:
            final_muts.extend(collapse_muts(muts))
    return final_muts


def label_mut_type(mut):
    """
    labels mutation as "deletion", "Insertion" or "substitution"
    :param mut: mutation to label
    :type mut: str
    :return: del, ins or sub
    :rtype: str
    """
    if mut.find('-') != -1:
        return 'del'
    if mut.find(':') != -1:
        return 'ins'
    return 'sub'
