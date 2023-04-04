import pandas as pd
import numpy as np
import json
import yaml


def combine_muts(row, cols, spike_only=True):
    """

    :param row:
    :type row:
    :param cols:
    :type cols:
    :return:
    :rtype:
    """
    all_muts = []
    for col in cols:
        if isinstance(row[col],float):
            continue
        muts = [x.strip() for x in row[col].split(',')]
        if spike_only:
            muts = [x for x in muts if x.startswith('S:')]
        all_muts.extend(muts)
    mut_str = (',').join(all_muts)
    return mut_str


def prep_nextclade_lineage_count():
    """
    takes results from nextclade alignment and prepares it to create constellation file
    :return:
    :rtype:
    """
    df = pd.read_csv('data/interim/nextclade_results/nextclade.tsv', sep='\t', header=0)
    df = df[(df['qc.overallStatus'] == 'good')]
    df['accession'] = df['seqName'].str.split(' ').str[0].str.strip()
    ndf = df.reset_index(drop=True)
    aa_cols = ['aaSubstitutions',
               'aaDeletions',
               'aaInsertions']
    ndf['aa_muts'] = ndf.apply(lambda x: combine_muts(x, aa_cols), axis=1)
    return ndf


def common_mutations(lineage, name, min_occurance=1, threshold_common=0.8, threshold_intermediate=0.2):
    rules = {
        'threshold_common':threshold_common,
        'threshold_intermediate':threshold_intermediate,
        'min_occurance':min_occurance
    }
    variants = [x for x in list(lineage.keys()) if x != 'total']
    total = lineage['total']
    common = []
    intermed = []
    low = []
    below_min_occur = []
    for variant in variants:
        n_var = lineage[variant]
        freq = float(n_var/ total)
        if freq > threshold_common:
            common.append(variant)
        elif freq > threshold_intermediate:
            intermed.append(variant)
        elif n_var > min_occurance:
            low.append("%s;%f" % (variant, freq))
        else:
            below_min_occur.append("%s;%f" % (variant, freq))
    meta = {'name': name,
            'total': lineage['total'],
            'common': common,
            'intermediate': intermed,
            'low': low,
            'below_min_occurance': below_min_occur,
            'rules':rules
           }
    return meta


def lineage_defining_muts(nextclade, min_occurance=1, threshold_common=0.8, threshold_intermediate=0.2):
    """

    :return:
    :rtype:
    """
    lineages_seq = nextclade.groupby('Nextclade_pango')['accession'].apply(list).to_dict()
    seq_muts = pd.Series(nextclade.aa_muts.values, index=nextclade.accession).to_dict()
    for k, v in seq_muts.items():
        if len(v) == 0:
            seq_muts[k] = []
        else:
            muts = v.split(',')
            seq_muts[k] = muts
    lineages_dict = {}
    for lineage, sequences in lineages_seq.items():
        for sequence in sequences:
            if lineage not in lineages_dict:
                lineages_dict[lineage] = {'total': 1}
            else:
                lineages_dict[lineage]['total'] += 1
            variants = seq_muts[sequence]
            for variant in variants:
                if variant in lineages_dict[lineage]:
                    lineages_dict[lineage][variant] += 1
                else:
                    lineages_dict[lineage][variant] = 1
    constellations = {}
    for name, lineage in lineages_dict.items():
        constell = common_mutations(lineage, name, min_occurance, threshold_common, threshold_intermediate)
        constellations[name] = constell
    with open('data/processed/misc/lineage_constellations.json', 'w') as outfile:
        json.dump(constellations, outfile, indent=4)
    return constellations


def define_lineage_inheritance(yml_path):
    """
    takes yml file from https://github.com/cov-lineages/lineages-website and converts to json
    where key = lineage name and value includes lineage children
    :param yml_path: path to yml file
    :type yml_path: str
    :return:
    :rtype:
    """
    with open(yml_path, 'r') as yaml_in, open("data/processed/misc/lineages_inheritance.json", "w") as json_out:
        yaml_object = yaml.safe_load(yaml_in)  # yaml_object will be a list or a dict
        json.dump(yaml_object, json_out)

    f = open("data/processed/misc/lineages_inheritance.json", )
    lineages = json.load(f)
    lineages_inheritance = {}
    for lineage in lineages:
        name = lineage['name']
        lineages_inheritance[name] = lineage
    with open("data/processed/misc/lineages_inheritance.json", 'w') as outfile:
        json.dump(lineages_inheritance, outfile, indent=4)
    return lineages_inheritance


