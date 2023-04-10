"""
Module to hold functions to help save/store/manipulate data structures
"""
import csv
import os
import pandas as pd
from itertools import islice
import pickle


def write_tsv_file(dataframe, save_path):
    """
    Function to save a pandas dataframe into tsv
    :param dataframe: dataframe to save
    :type dataframe: pandas.DataFrame
    :param save_path: path to save the TSV
    :type save_path: str
    :return: just saves the file
    :rtype: None
    """
    with open(save_path, 'wt') as f:
        tbl = csv.writer(f, delimiter='\t', lineterminator=os.linesep)
        tbl.writerows(dataframe)


def read_tsv_file(path):
    """
    function to read tsv file and return a dataframe
    :param path: save path to read
    :type path: str
    :return: dataframe
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(path, sep='\t', header=0)
    return df


def combine_seq_dicts(file_list):
    """
    takes a list of saved dictionary paths and combines them into 1 dictionary
    :param file_list: list of saved dictionaries
    :type file_list: list
    :return: combined dictionary
    :rtype: dict
    """
    seq_dict = {}
    for file in file_list:
        with open(file, 'rb') as f:
            values = pickle.load(f)
        seq_dict.update(values)
    return seq_dict


def folder_file_list(folder, suffix):
    """
    get a list of file paths in a folder if they end with a certain suffix
    :param folder: folder to search
    :type folder: str
    :param suffix: suffix to check for
    :type suffix: str
    :return: file list
    :rtype: list
    """
    save_files = []
    # if os.path.isfile(self.seq_change_path):
    #    change_save_files.append(self.seq_change_path)
    for file in os.listdir(folder):
        if file.endswith(suffix):
            save_files.append(folder + '/' + file)
    return save_files


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def chunk_it(data, size):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def chunk_dictionary(data, size):
    return list(chunk_it(data, size))


def rename_multicol_df(df):
    """
    function to rename a multi level column dataframe as a combination of the two levels
    :param df: dataframe to fix
    :type df: pandas.DataFrame
    :return: renamed dataframe
    :rtype: pandas.DataFrame
    """
    col_names = []
    for tup in df.columns:
        if len(tup[1]) > 0:
            col_name = ('_').join(tup)
        else:
            col_name = tup[0]
        col_names.append(col_name)
    df.columns = col_names
    return df


def check_directory(path):
    """
    checks if path (folder) exists. If it doesn't, makes one

    :param path: folder path to check
    :type path: str
    :return:
    :rtype:
    """
    if not os.path.exists(path):
        os.makedirs(path)
