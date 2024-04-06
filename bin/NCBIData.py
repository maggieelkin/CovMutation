from Bio import SeqIO
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from DataHelpers import read_tsv_file
from Bio.SeqRecord import SeqRecord
from MutationHelpers import translate_seq, check_known_aa, pullout_pos_mut_df, label_mut_type
import pickle
from datetime import datetime
from collections import Counter


def write_to_filter_notes(msg, append=True):
    """
    function to save filtering message to a text file
    :param msg: message to write to the text file
    :type msg: str
    :param append: if true, append to file, otherwise overwrite file
    :type append: bool
    :return: None
    :rtype:
    """
    text_path = 'data/MetaData/metadata_filtering_info.txt'
    if append:
        method = '+a'
    else:
        method = 'w'
    date = datetime.now().strftime("%b %d %Y %H:%M:%S")
    write_msg = date + ": " + msg
    with open(text_path, method) as f:
        f.write(write_msg)


def get_genome_table(json_line, data=[]):
    """
    pulls out the gene annotations from data_report from NCBI virus dataset

    :param json_line: line from jsonl file, dictionary of isolate meta data
    :type json_line: dict
    :param data: current list to append data to
    :type data: list
    :return: list of genome table data
    :rtype: list
    """
    annot = json_line['annotation']
    for gene in annot.get('genes', []):
        for c in gene.get('cds', []):
            ranges = []
            for r in c['nucleotide']['range']:
                ranges.append([int(r['begin']), int(r['end'])])
            prot_len = int(c['protein']['range'][-1]['end'])
            genome_table = {'gene_name': gene['name'],
                            'nuc_acc': c['nucleotide']['accessionVersion'],
                            'nuc_ranges': ranges,
                            'prot_name': c['name'],
                            'prot_acc': c['protein']['accessionVersion'],
                            'prot_len': prot_len
                            }
            data.append(genome_table)
    return data


def read_data_report(json_file_path):
    """
    function to read in the jasonl data report file to return genome table (with all genes) and meta_data table

    :param json_file_path: filepath that contains the data report in jasonl format
    :type json_file_path: str
    :return: genome_table and meta_data
    :rtype: tuple
    """
    meta_data_list = []
    genome_table_list = []
    for line in open(json_file_path, 'r'):
        g = json.loads(line)
        if 'annotation' in g:
            genome_table_list = get_genome_table(g, data=genome_table_list)
            try:
                nuc_complete = g['nucleotideCompleteness']
            except KeyError:
                nuc_complete = g['completeness']
            try:
                pangolin = g['virus']['pangolinClassification']
            except KeyError:
                pangolin = np.nan
            try:
                data = {
                    'Nucleotide Accession': g['accession'],
                    'Species Taxonomy Id': g['virus']['taxId'],
                    'Species Name': g['virus']['sciName'],
                    'Isolate Name': g['isolate']['name'],
                    'Nucleotide Length': g['length'],
                    'Sequence Type': g['sourceDatabase'],
                    'Nuc Completeness': nuc_complete,
                    'Geo Location': "{}; {}".format(g['location']['geographicRegion'],
                                                    g['location']['geographicLocation']),
                    'Host Taxonomy ID': g['host']['taxId'],
                    'Collection Date': g['isolate']['collectionDate'],
                    'Release Date': g['releaseDate'],
                    'pangolin': pangolin
                }
                meta_data_list.append(data)
            except KeyError:
                pass
    meta_data = pd.DataFrame.from_records(meta_data_list)
    genome_table = pd.DataFrame.from_records(genome_table_list)
    return meta_data, genome_table


def get_full_metadata(main_folder='data/external/ncbi', save_folder='data/MetaData', spike_names=['S']):
    """
    function to read in data reports, filtering out sequences that are not annotated.
    saves the genome info to metadata folder

    :param spike_names: list of gene names to filter for spike gene
    :type spike_names: list
    :param save_folder: folder to save genome info and meta data default is data/MetaData
    :type save_folder: str
    :param main_folder: folder that holds subfolders of different downloads from NCBI, data from different dates and ref
    :type main_folder: string
    :return: returns dataframe of sequence meta_data and saves it
    :rtype: pandas.DataFrame
    """
    # read in data_report files
    json_file_paths = []
    for folder in os.listdir(main_folder):
        json_file_path = main_folder + '//' + folder + '//data_report.jsonl'
        json_file_paths.append(json_file_path)
    meta_data = pd.DataFrame()
    genome_info = pd.DataFrame()
    for i in tqdm(range(len(json_file_paths)), desc='data reports'):
        # print(json_file_paths[i])
        meta, genome = read_data_report(json_file_paths[i])
        genome_info = genome_info.append(genome)
        meta_data = meta_data.append(meta)

    genome_info = genome_info.drop_duplicates(['gene_name', 'nuc_acc', 'prot_name', 'prot_acc', 'prot_len'],
                                              keep='last')
    genome_info = genome_info.reset_index(drop=True)
    genome_info.to_pickle(save_folder + "//ncbi_virus_genome_info.pkl")

    meta_data = meta_data.drop_duplicates()
    # join the Spike Annotation to the meta_data
    sdf = genome_info[(genome_info['gene_name'].isin(spike_names))]
    meta_data = pd.merge(meta_data, sdf, left_on='Nucleotide Accession', right_on='nuc_acc')
    # need to fix the nuc_ranges to just get a single level list
    meta_data['nuc_ranges'] = meta_data['nuc_ranges'].apply(lambda x: x[0])
    meta_data = meta_data.reset_index(drop=True)
    msg = "Total Number Annotated Accessions: {} \n".format(len(meta_data))
    print(msg)
    write_to_filter_notes(msg, append=False)
    meta_data.to_pickle(save_folder + "//sequences_annot_metadata.pkl")
    return meta_data


def remove_earlier_accessions(meta_data):
    """
    accessions can have different versions. This will keep the latest version
    version number is after accession (".xx")

    :param meta_data:
    :type meta_data:
    :return:
    :rtype:
    """
    # remove the reference record to ensure the one we want is included
    meta_data[['acc_base', 'version']] = meta_data['Nucleotide Accession'].str.split('.', expand=True)
    ref = meta_data[(meta_data['Nucleotide Accession'] == 'NC_045512.2')]
    meta_data = meta_data[~(meta_data['Nucleotide Accession'].str.contains('NC_045512'))]
    meta_data = meta_data.sort_values(['acc_base', 'version'])
    meta_data = meta_data.drop_duplicates(subset='acc_base', keep='last')
    meta_data = ref.append(meta_data)
    meta_data = meta_data.reset_index(drop=True)
    return meta_data


def initial_metadata_filtering(meta_data=None):
    """
    filters records based on meta data alone
    removes earlier accessions
    pulls out region and country from geo location
    fixes reference record
    removes records with ambiguous dates

    :param meta_data: Full output data from get_full_metadata
    :type meta_data: pandas.DataFrame
    :return: filtered metadata based on date
    :rtype: pandas.DataFrame
    """
    if meta_data is None:
        df = pd.read_pickle('data/MetaData/sequences_annot_metadata.pkl')
    else:
        df = meta_data
    df = df[~(df['Isolate Name'].isnull())]
    df = df[~(df['Collection Date'].isnull())]
    msg = "Total Number Accessions after removing null date or null isolate name: {} \n".format(len(df))
    print(msg)
    df = remove_earlier_accessions(df)
    msg1 = "Total Number Accessions after removing earlier versions: {} \n".format(len(df))
    print(msg1)
    df = df.reset_index(drop=True)
    # pull out country
    df[['region', 'country']] = df['Geo Location'].str.split(';', expand=True)
    df[['country', 'division']] = df['country'].str.split(':', expand=True)
    df['country'] = df['country'].str.strip()
    # fix reference meta_data with proper isolate name ('Wuhan/Hu-1/2019') and date
    i = df[(df['Nucleotide Accession'] == 'NC_045512.2')].index[0]
    name_index = df.columns.tolist().index('Isolate Name')
    date_index = df.columns.tolist().index('Collection Date')
    df.iloc[i, name_index] = 'Wuhan/Hu-1/2019'
    df.iloc[i, date_index] = '2019-12-26'
    # remove dates that are incomplete
    df = df[~(df['Collection Date'].str.count('-') < 2)]
    msg2 = "Total Number Accessions after removing ambiguous dates: {} \n".format(len(df))
    print(msg2)
    # collection date to date time (removes ambigous dates)
    df['Collection Date'] = pd.to_datetime(df['Collection Date'])
    df['Release Date'] = pd.to_datetime(df['Release Date'])
    df['collection_release_diff'] = df['Release Date'] - df['Collection Date']
    df['collection_release_diff'] = df['collection_release_diff'].dt.days
    df = df[(df['collection_release_diff'] <= 60)]
    df = df.reset_index(drop=True)
    msg3 = "Total Number Accessions after removing late submission: {} \n".format(len(df))
    print(msg3)
    msg = msg + msg1 + msg2 + msg3
    df.to_pickle("data/MetaData/sequences_full_metadata.pkl")
    write_to_filter_notes(msg, append=True)
    return df


def filter_record_quality(records, N_perc=.05, min_length=29000):
    """
    filters fasta records. Keeps records if they have %N < N_perc and length >= min_length
    :param records: list of fasta records
    :type records: list
    :param N_perc: maximum %N content, default is 1
    :type N_perc: float
    :param min_length: minimum length of nucleotide sequence, default is 29,000
    :type min_length: int
    :return: dictionary of key = record.id, value = record
    :rtype: dict
    """
    good_seqs = {}
    for i in tqdm(range(len(records)), desc='FASTAs'):
        record = records[i]
        sequence = record.seq
        n = sequence.count('N')
        length = len(sequence)
        perc_n = (n / length) * 100
        if length >= min_length and perc_n < N_perc:
            good_seqs[record.id] = record
    print("total number of sequnces passed filter: " + str(len(good_seqs)))
    return good_seqs


def remove_redundant_seq_collection(meta_data, records, per_month=True):
    """
    function takes the full sequences for nextstrain and removes same ones per country, year, month
    needs to take the sequences out of data/raw/sequences_ncbi so the seq.id is the accession
    """
    if isinstance(records, list):
        record_seq = {}
        for record in records:
            record_seq[record.id] = str(record.seq)
    else:
        record_seq = {}
        for record_id, record in records.items():
            record_seq[record_id] = str(record.seq)
    # remove the reference record to ensure the one we want is included
    ref = meta_data[(meta_data['Nucleotide Accession'] == 'NC_045512.2')]
    df = meta_data[~(meta_data['Nucleotide Accession'] == 'NC_045512.2')]
    df['seq'] = df['Nucleotide Accession'].map(record_seq)
    df = df.sort_values('Collection Date')
    if per_month:
        df['collection year'] = df['Collection Date'].dt.year
        df['collection month'] = df['Collection Date'].dt.month
        df = df.drop_duplicates(['country', 'collection year', 'collection month', 'seq'])
    else:
        df = df.drop_duplicates(['country', 'Collection Date', 'seq'])
    df.drop('seq', inplace=True, axis=1)
    df = ref.append(df)
    df = df.reset_index(drop=True)
    return df


def filter_sequences(meta_data=None, main_folder='data/external/ncbi', remove_redunant=True, **kwargs):
    """
    function takes the sequences list from meta_data and genomic FASTA files in main_folder and creates a fasta file
    of filtered sequence
    :param remove_redunant: If True, remove same NT sequences per country, year, month
    :type remove_redunant: bool
    :param meta_data: dataframe of meta data
    :type meta_data: pandas.DataFrame
    :param main_folder: folder that holds subfolders of genomic fasta files
    :type main_folder: str
    :return: fasta file of filtered sequences
    :rtype: fasta
    """
    if meta_data is None:
        df = pd.read_pickle('data/MetaData/sequences_full_metadata.pkl')
    else:
        df = meta_data
    # fasta paths to read in
    fasta_paths = []
    for folder in os.listdir(main_folder):
        fasta_path = main_folder + '//' + folder + '//genomic.fna'
        fasta_paths.append(fasta_path)
    # get records in dictionary
    all_records = {}
    for i in tqdm(range(len(fasta_paths)), desc='FASTA files'):
        fasta_path = fasta_paths[i]
        # for record in SeqIO.parse(fasta_path, 'fasta'):
        #    all_records[record.id] = record
        record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))
        all_records.update(record_dict)
    del record_dict

    if remove_redunant:
        print("Total Number Accessions before removing redundant: {}".format(len(df)))
        df = remove_redundant_seq_collection(df, all_records)
        msg = "Total Number Accessions after removing redundant: {} \n".format(len(df))
        print(msg)

    accessions = list(set(df['Nucleotide Accession'].values))
    print("Total Number Accessions: {}".format(len(accessions)))
    print("total number of sequences prior to any filtering: " + str(len(all_records)))
    all_records = {k: all_records[k] for k in accessions}
    print("total number of sequences in meta data: " + str(len(all_records)))
    all_records = list(all_records.values())
    # record quality filter
    seqs_pass_filter = filter_record_quality(all_records, **kwargs)
    msg1 = "total number of sequences After Filtering for sequence quality: {} \n".format(len(seqs_pass_filter))
    print(msg1)
    del all_records
    # filter the df from the last filter
    good_accessions = list(seqs_pass_filter.keys())
    if 'NC_045512.2' not in good_accessions:
        good_accessions.insert(0, 'NC_045512.2')
    df = df[(df['Nucleotide Accession'].isin(good_accessions))]
    msg2 = "total number of sequences in metadata After Filtering for sequence quality: {} \n".format(len(df))
    print(msg2)
    # spike sequence translation filter
    records = list(seqs_pass_filter.values())
    acc_pass_spike_translation = translate_spike_filter(records=records, meta_data=df)

    good_accessions = list(acc_pass_spike_translation.keys())
    seqs_pass_filter = {k: seqs_pass_filter[k] for k in good_accessions}
    records = list(seqs_pass_filter.values())
    msg3 = "total number of sequences After Filtering for spike seq translation: {} \n".format(len(seqs_pass_filter))
    print(msg3)
    # need to make sure that this reference record is not filtered out for any reason
    if 'NC_045512.2' not in good_accessions:
        good_accessions.insert(0, 'NC_045512.2')
    df = df[(df['Nucleotide Accession'].isin(good_accessions))]
    df = df.reset_index(drop=True)
    msg4 = "total number of sequences in metadata After Filtering for seq Translation: {} \n".format((len(df)))
    print(msg4)
    df.to_pickle('data/MetaData/sequences_filtered_metadata.pkl')
    print("Total Number Accessions After Filtering Steps: {}".format(len(df)))
    print("total number of sequences After Filtering Steps: " + str(len(records)))
    msg = msg + msg1 + msg2 + msg3 + msg4
    write_to_filter_notes(msg)
    SeqIO.write(records, "data/raw/sequences_ncbi.fasta", "fasta")
    print('Done')
    return records


def translate_spike_filter(records, meta_data=None):
    """
    from full_metadata, pull out Accession, nuc_ranges (for spike seq), isolate name, collection date.
    This will form a meta_data dict that also holds translated spike sequences
    from list of records, translate the spike sequences, filter out those that have unknown (ambiguous AA)
    save the meta_data dict
    :param records: list of fasta records
    :type records: list
    :param meta_data: dataframe of metadata for sequences
    :type meta_data: pandas.DataFrame
    :return: dict of key=accession, value = metadata of sequences that passed translation filter (known AA)
    :rtype: dict
    """
    if meta_data is None:
        df = pd.read_pickle('data/MetaData/sequences_full_metadata.pkl')
    else:
        df = meta_data
    df = df[['Nucleotide Accession', 'nuc_ranges', 'Isolate Name', 'Collection Date']]
    acc_meta_data = df.set_index('Nucleotide Accession').to_dict('index')

    # maybe have to update the reference one because it got messed up somewhere upstream from this
    # acc_range['NC_045512.2']['nuc_ranges'] = [21563, 25384]
    excl = []
    for record in tqdm(records):
        meta = acc_meta_data[record.id]
        nuc_range = meta['nuc_ranges']
        spike_seq = translate_seq(record.seq, nuc_range)
        if isinstance(spike_seq, str) and check_known_aa(spike_seq):
            meta['spike_seq'] = spike_seq
        else:
            excl.append(record.id)
    for key in excl:
        acc_meta_data.pop(key)
    with open('data/interim/sequences_trans_spike.pkl', 'wb') as f:
        pickle.dump(acc_meta_data, f)
    return acc_meta_data


def filter_nextclade_qc(meta_data=None, bad_ids=None):
    """
    function takes sequences_full_metadata, fixes issues with reference seq for nextstrain
    reads nextclade.tsv for good quality sequences 
    removes bad sequences, sequences with ambiguous dates and specific bad_ids
    end dataframe has final sequences that pass for analysis
    :param meta_data: metadata of sequences, if nothing is passed, take the designated saved path
    :type meta_data: pandas.DataFrame
    :param bad_ids: list of ids to exclude
    :type bad_ids: list
    :return: filtered meta data
    :rtype: pandas.DataFrame
    """
    if bad_ids is None:
        bad_ids = []
    if meta_data is None:
        df = pd.read_pickle('data/MetaData/sequences_filtered_metadata.pkl')
    else:
        df = meta_data
    print("Total Records before filtering for nextclade QC: {}".format(len(df)))
    # ones to exclude because they fuck up nextstrain or have very short protein length (last one)
    exclude = ['SARS-CoV-2/human/ECU/Z&Z_SARS_1/2020',
               'SARS-CoV-2/human/ECU/Z&Z_SARS_4/2020',
               'SARS-CoV-2/human/ECU/Z&Z_SARS_3/2020',
               'SARS-CoV-2/human/ECU/Z&Z_SARS_2/2020',
               'hCoV-19/Bangladesh/ECHD_00716/2020',
               'SARS-CoV-2/human/FRA/IHUMI-6070VR_10^-5/2022',
               'SARS-CoV-2/human/FRA/IHUMI-6070VR_10^-4/2022']
    for bad_id in bad_ids:
        exclude.append(bad_id)
    df = df[~(df['Isolate Name'].isin(exclude))]
    # get bad QC accessions from nextclade
    ndf = read_tsv_file('data/interim/nextclade_results/nextclade.tsv')
    ndf['accession'] = ndf['seqName'].str.split(' ').str[0].str.strip()
    good_accessions = ndf[(ndf['qc.overallStatus'] == 'good')]['accession'].values.tolist()
    if 'NC_045512.2' not in good_accessions:
        good_accessions.insert(0, 'NC_045512.2')
    df = df[(df['Nucleotide Accession'].isin(good_accessions))]
    msg = "Number Sequences passed Nextclade QC: {} \n".format(len(df))
    print(msg)

    nextclade = ndf[['accession', 'clade', 'Nextclade_pango', 'substitutions', 'deletions', 'insertions', 'frameShifts',
                     'aaSubstitutions', 'aaDeletions', 'aaInsertions']]
    df = pd.merge(df, nextclade, left_on='Nucleotide Accession', right_on='accession')
    aa_col = ['aaSubstitutions', 'aaDeletions', 'aaInsertions']
    for col in aa_col:
        df[col + '_spike'] = df[col].apply(lambda x: spike_muts_only(x))
    df['spike_mutations'] = df['aaSubstitutions_spike'] + df['aaDeletions_spike'] + df['aaInsertions_spike']
    df = df.reset_index(drop=True)
    df.to_pickle('data/MetaData/sequences_metadata_final.pkl')
    write_to_filter_notes(msg)
    get_final_filtered_seq(meta_data=df)
    # filter sequences to those in metadata_nextclade
    ids = list(df['Nucleotide Accession'].values)
    print("Filtering Fastas...")
    fasta_path = "data/raw/sequences_ncbi.fasta"
    records = SeqIO.parse(fasta_path, 'fasta')
    records = (record for record in records if record.id in ids)
    print("Saving new fasta")
    SeqIO.write(records, "data/raw/sequences_ncbi_final.fasta", "fasta")
    print("Done")


def get_final_filtered_seq(meta_data=None):
    """
    After filtering from Nextclade QC, takes the spike sequences that passed QC and saves in processed data folder

    :param meta_data: meta data for final filtered sequences (after nextclade QC)
    :type meta_data: pandas.DataFrame
    """
    if meta_data is not None:
        df = meta_data
    else:
        df = pd.read_pickle('data/MetaData/sequences_metadata_final.pkl')
    msg = 'Total number of sequences in final dataset: {} \n'.format(len(df))
    print(msg)
    with open('data/interim/sequences_trans_spike.pkl', 'rb') as f:
        seq_data = pickle.load(f)
    acc = df['Nucleotide Accession'].values.tolist()
    seq_data = {k: seq_data[k] for k in acc}
    msg1 = 'Total number of sequences in final seq_data: {} \n'.format(len(seq_data))
    print(msg1)
    with open('data/processed/full_data/sequences_trans_spike.pkl', 'wb') as f:
        pickle.dump(seq_data, f)
    print('getting and saving unique sequences metadata')
    df = unique_seq_metadata(seq_data, meta_data=df)
    df = df.reset_index(drop=True)
    msg2 = 'Total number of sequences with unique spike in final dataset: {} \n'.format(len(df))
    print(msg2)
    df.to_pickle('data/MetaData/sequences_metadata_final_unique.pkl')
    msg = msg + msg1 + msg2
    write_to_filter_notes(msg)


def nextstrain_meta_data_prep(df):
    """
    function renames columns in meta_data for nextstrain
    :param df: meta data dataframe
    :type df: pandas.DataFrame
    :return: nextstrain ready dataframe
    :rtype: pandas.DataFrame
    """
    df = df.reset_index(drop=True)
    i = df[(df['Nucleotide Accession'] == 'NC_045512.2')].index[0]
    name_index = df.columns.tolist().index('Isolate Name')
    date_index = df.columns.tolist().index('Collection Date')
    df.iloc[i, name_index] = 'Wuhan/Hu-1/2019'
    df.iloc[i, date_index] = '2019-12-26'
    df['gisaid_epi_isl'] = '?'
    df['region_exposure'] = df['region']
    df['country_exposure'] = df['country']
    df['division_exposure'] = df['division']
    df['segment'] = 'genome'
    df['Virus Genus'] = 'Betacoronavirus'
    df['Host Name'] = "Human"
    df = df[
        ['Isolate Name', 'Virus Genus', 'gisaid_epi_isl', 'Nucleotide Accession', 'Collection Date', 'region',
         'country', 'division', 'region_exposure', 'country_exposure', 'division_exposure', 'Nucleotide Length',
         'Host Name', 'Release Date']]
    df.columns = ['strain', 'virus', 'gisaid_epi_isl', 'genbank_accession', 'date', 'region', 'country', 'division',
                  'region_exposure', 'country_exposure', 'division_exposure', 'length', 'host', 'date_submitted']
    df = df.fillna('?')
    return df


def prep_nextstrain(meta_data=None):
    """
    Preps data from final filtered sequences for nextstrain
    :param meta_data: metadata output from filter_nextclade_qc
    :type meta_data: pandas.DataFrame
    """
    if meta_data is None:
        df = pd.read_pickle('data/MetaData/sequences_metadata_final.pkl')
    else:
        df = meta_data
    with open('data/processed/full_data/sequences_trans_spike.pkl', 'rb') as f:
        seq_data = pickle.load(f)
    # get unique spike sequences

    seq_data = get_unique_seq_data(seq_data)
    acc = list(seq_data.keys())
    ref_acc = ['MN908947.3', 'NC_045512.2']
    acc = ref_acc + acc
    df = df[(df['Nucleotide Accession'].isin(acc))]
    df = df.reset_index(drop=True)
    id_dict = dict(zip(df['Nucleotide Accession'], df['Isolate Name']))
    fasta_path = 'data/raw/sequences_ncbi_final.fasta'
    records = SeqIO.to_dict(SeqIO.parse(fasta_path, 'fasta'))
    print("total fasta records: {}".format(len(records)))
    records = {k: records[k] for k in id_dict}
    records = list(records.values())
    print("total unique spike fasta records: {}".format(len(records)))

    # rename records with isolate name for nextstrain
    print("total fasta records to rebuild for nextstrain: {}".format(len(records)))
    good_records = []
    for record in records:
        name = id_dict[record.id]
        new_record = SeqRecord(record.seq, id=name, name=name, description=name)
        good_records.append(new_record)
    SeqIO.write(good_records, "data/interim/nextstrain_prep/sequences.fasta", "fasta")
    print('Number of Sequences: {}'.format(len(good_records)))
    print('Number in meta data: {}'.format(len(df)))
    del good_records
    # finish meta data processing
    df = nextstrain_meta_data_prep(df)
    df.to_csv('data/interim/nextstrain_prep/sequences_metadata.tsv', sep='\t', index=False)


def spike_muts_only(muts):
    if not isinstance(muts, str):
        return []
    mut_list = muts.split(',')
    mut_list = [x.strip() for x in mut_list]
    mut_list = [x.strip() for x in mut_list if x.startswith('S:')]
    mut_list = [x.replace('S:', '').strip() for x in mut_list]
    return mut_list


def weighted_subsample_train_test(pre_cutoff_seq, size=10000, save_path='data/processed/full_data/train_test_seq.pkl'):
    """
    takes a weighted subsample of pre_cutoff sequences of a specified size, with no replacement.
    weights are determined by the absolute count of the sequences in dataset
    :param pre_cutoff_seq: list of possible training sequences (pre cut off), values in list are strings
    :type pre_cutoff_seq: list
    :param size: size of training dataset to get subsample of, default is 10,000
    :type size: int
    :param save_path: path to pickle dump train/test dictionary
    :type save_path: str
    :return: dict of train and test lists
    :rtype: dict
    """
    print('Number precutoff prior to subsample: {}'.format(len(pre_cutoff_seq)))
    cnt = dict(Counter(pre_cutoff_seq))
    seq = list(cnt.keys())
    print('Number unique precutoff prior to subsample: {}'.format(len(seq)))
    abs_cnt = list(cnt.values())
    n = len(pre_cutoff_seq)
    weights = []
    for x in abs_cnt:
        w = x / n
        weights.append(w)
    train = np.random.choice(seq, size=size, replace=False, p=weights)
    assert len(train) == len(set(train))
    test = [x for x in seq if x not in train]
    assert len(test) == len(set(test))
    train_test = {
        'train': train,
        'test': test
    }
    with open(save_path, 'wb') as f:
        pickle.dump(train_test, f)
    return train_test


def subset_train_seq(pre_cutoff_seq, save_path='data/processed/full_data/train_seq_subbed.pkl'):
    print('Number precutoff prior to subsample: {}'.format(len(pre_cutoff_seq)))
    cnt_seq = dict(Counter(pre_cutoff_seq))
    print('Number unique precutoff prior to subsample: {}'.format(len(cnt_seq)))
    subbed_seq = []
    for k, v in cnt_seq.items():
        if v > 1:
            subbed_seq.append(k)
    print('Number Subsampled Sequences with frequency > 1: {}'.format(len(subbed_seq)))
    with open(save_path, 'wb') as f:
        pickle.dump(subbed_seq, f)
    return subbed_seq


def full_sequences_training(cut_off='2022-1-1', subsample_size=10000, **kwargs):
    """
    gets training sequences prepared
    saves unique training sequences -> default 'data/processed/full_data/train_seq.pkl'
    subsamples those and saves those -> default 'data/processed/full_data/train_test_seq.pkl', can change with kwargs

    :param cut_off: cutoff date in %Y-%m-%d format, sequences before cutoff are used for training, default is 2022-1-1
    :type cut_off: str
    :param subsample_size: size of training dataset to get subsample of, default is 10,000
    :type subsample_size: int
    :return: nothing, just saves unique training seq and text file with size counts
    :rtype: None
    """
    # TODO: Refractor training function to add a column in metadata_final_unique that marks in training dataset
    cut_off_datetime = datetime.strptime(cut_off, '%Y-%m-%d')
    with open('data/processed/full_data/sequences_trans_spike.pkl', 'rb') as f:
        seq_data = pickle.load(f)
    print('Total # sequences: {}'.format(len(seq_data)))
    pre_cutoff_train = []
    for accession, meta in seq_data.items():
        date = meta['Collection Date']
        if date < cut_off_datetime:
            pre_cutoff_train.append(meta['spike_seq'])
    n_seq_total = len(pre_cutoff_train)
    set_pre_cutoff_train = list(set(pre_cutoff_train))
    n_unique = len(set_pre_cutoff_train)
    print('getting subsampled sequences')
    train_test_dict = weighted_subsample_train_test(pre_cutoff_train, size=subsample_size, **kwargs)
    n_subsampled_train = len(train_test_dict['train'])
    n_subsampled_test = len(train_test_dict['test'])

    train_seq_subbed = subset_train_seq(pre_cutoff_train)
    n_train_seq_subbed = len(train_seq_subbed)

    message = "Cut-off:{} \nNumber Pre Cut-off Seq:{} " \
              "\nNumber Unique Training Seq:{} " \
              "\nNumber Subbed Training Seq by frequency:{} " \
              "\nNumber Sub-sampled Training Seq:{} " \
              "\nNumber Sub-sampled Test Seq:{} ".format(cut_off, n_seq_total, n_unique, n_train_seq_subbed,
                                                         n_subsampled_train, n_subsampled_test)
    print(message)
    with open('data/processed/full_data/train_seq_info.txt', 'w') as f:
        f.write(message)
    with open('data/processed/full_data/train_seq.pkl', 'wb') as f:
        pickle.dump(set_pre_cutoff_train, f)


def get_unique_seq_data(seq_data):
    """

    :param seq_data:
    :type seq_data:
    :return:
    :rtype:
    """
    unique = {}
    for k, v in seq_data.items():
        seq = v['spike_seq']
        if seq not in unique:
            unique[seq] = {
                'accession': k,
                'Collection Date': v['Collection Date']
            }
        else:
            if v['Collection Date'] < unique[seq]['Collection Date']:
                unique[seq] = {
                    'accession': k,
                    'Collection Date': v['Collection Date']
                }
    unique_acc = []
    for k, v in unique.items():
        unique_acc.append(v['accession'])
    unique_seq_data = {k: seq_data[k] for k in unique_acc}
    return unique_seq_data


def unique_seq_metadata(seq_data, meta_data):
    """
    calls get_unique_seq_data on the seq_data dict (holds spike sequences)
    filters meta_data df to just those accessions
    :param seq_data:
    :type seq_data:
    :param meta_data:
    :type meta_data:
    :return:
    :rtype:
    """
    print('Number of sequences: {}'.format(len(seq_data)))

    print('Getting Unique Sequences Only')
    seq_data = get_unique_seq_data(seq_data)
    print('Number of unique sequences: {}'.format(len(seq_data)))
    df = meta_data[(meta_data['Nucleotide Accession'].isin(seq_data))]
    df = df.reset_index(drop=True)
    print('Number of unique sequences in meta_data: {}'.format(len(df)))
    return df


def mutation_summary_full_data(mut_save_path="data/processed/full_data/mutts.pkl", unique=True):
    """

    :return:
    :rtype:
    """
    if unique:
        df = pd.read_pickle('data/MetaData/sequences_metadata_final_unique.pkl')

    else:
        df = pd.read_pickle('data/MetaData/sequences_metadata_final.pkl')
    print('Number of sequences in meta data: {}'.format(len(df)))
    df = df[['Nucleotide Accession', 'Collection Date', 'spike_mutations']]
    acc_muts = df.set_index('Nucleotide Accession').to_dict('index')
    mut_dict = {}
    for accession, seq_meta in acc_muts.items():
        mutations = seq_meta['spike_mutations']
        if isinstance(mutations, float):
            continue
        for mut in mutations:
            mut_type = label_mut_type(mut)
            if mut not in mut_dict:
                first_date = seq_meta['Collection Date']
                mut_dict[mut] = {'first_date': first_date, 'n_times': 1, 'mut_type': mut_type}
            else:
                meta = mut_dict[mut]
                date = seq_meta['Collection Date']
                if date < meta['first_date']:
                    meta['first_date'] = date
                meta['n_times'] = meta['n_times'] + 1
    mut_df = pd.DataFrame.from_dict(mut_dict).T.reset_index()
    mut_df.columns = ['mutation', 'first_date', 'n_times', 'mut_type']
    mut_df[['wt', 'pos', 'mut']] = mut_df.apply(lambda x: pullout_pos_mut_df(x), axis=1, result_type='expand')
    mut_df.to_pickle(mut_save_path)
    print('Number unique Mutations: {}'.format(len(mut_df)))
    return mut_df


if __name__ == '__main__':
    print('getting full metadata')
    get_full_metadata()

    print('initial metadata filtering')
    initial_metadata_filtering()

    print('filtering sequences for quality')
    filter_sequences()

    # inbetween these two steps Nextclade needs to be run and results saved in data/interim/nextclade_results

    print('filtering sequences for nextclade QC')
    filter_nextclade_qc()

    print('getting full sequences training')
    full_sequences_training()

    print('nextstrain prep')
    prep_nextstrain()

    print('mutation summary')
    mut_df = mutation_summary_full_data()
