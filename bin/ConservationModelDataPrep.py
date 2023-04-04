import pickle
from MutationHelpers import *
import scipy.stats as ss
from BioTransLanguageModel import *

pd.options.mode.chained_assignment = None


def nlargest_prob_entropy(df, nlargest):
    """
    for each position, get the nlargest probabilities and find the entropy of them
    if nlargest = 20, then its entropy for all AA probabilities for each position
    :param df: sequence probability dataframe,requires position to be called "pos" and biotrans prob to be called"prob"
    :type df: pandas.DataFrame
    :param nlargest: number of largest probability values to pull for each position (this can include WT)
    :type nlargest: int
    :return: dict of key = position, value = entropy
    :rtype: dict
    """
    top = pd.DataFrame(df.groupby('pos')['prob'].nlargest(nlargest).reset_index(level=1, drop=True))
    top_entropy = top.groupby('pos')['prob'].apply(ss.entropy).to_dict()
    return top_entropy


def top_proba_feat(df):
    """
    create features number top_1, top_2, ... top_20. Represents the 20 posterior probabilities sorted for each position

    :param df: sequence probability dataframe,requires position to be called "pos" and biotrans prob to be called"prob"
    :type df: pandas.DataFrame
    :return: top_n features in dictionary form (key = feature name, value = dict of key=position, value = value)
    :rtype: dict
    """
    top = pd.DataFrame(df.groupby('pos')['prob'].nlargest(20).reset_index(level=1, drop=True))
    top['top_n'] = top.groupby(['pos']).cumcount() + 1
    top = pd.pivot_table(top, values='prob', index='pos', columns='top_n')
    col = top.columns.tolist()
    col = ['top_{}'.format(x) for x in col]
    top.columns = col
    top_n = top.to_dict()
    return top_n


def wt_prob_from_df(df):
    """
    takes a dataframe built from biotransformer probabilites, pulls the wt probability into a dictionary
    dict can later be used as a map for a position dataframe
    :param df: sequence probability dataframe,requires the current AA to be called "WT"
    :type df: pandas.DataFrame
    :return: dict of key=position, value = current AA probability
    :rtype: dict
    """
    wt_df = df[(df['mutation'] == 'WT')]
    wt_pos_prob = pd.Series(wt_df.prob.values, index=wt_df.pos).to_dict()
    return wt_pos_prob


def hyphy_pipeline_features(hyphy_data, groups):
    """
    takes result of the hyphy pipeline and creates position based dictionaries for feature mapping

    :param hyphy_data: hyphy pipeline results are formatted in a dictionary of key=position,
    value = calculated statistics for position
    :type hyphy_data: dict
    :param groups: list of group headers to include
    :type groups: list
    :return: dictionary of key = feature name, value = position dictionary (key = pos, value=value)
    :rtype: dict
    """
    features_dict = {}
    for pos, pos_data in hyphy_data.items():
        for group, group_dict in pos_data.items():
            if group in groups:
                for group_val, val in group_dict.items():
                    feature = "{}_{}".format(group, group_val)
                    if feature not in features_dict:
                        features_dict[feature] = {}
                    features_dict[feature][pos] = val
    return features_dict


def position_embedding_features(seq, model_path=None):
    """
    gets embedding vector for each position to use in model
    :param seq: sequence of interest
    :type seq: str
    :param model_path: if provided, load pretrained biotransformer
    :type model_path: str
    :return: dataframe of embedding vector for each position (position is index)
    :rtype: pandas.DataFrame
    """
    bio_trans = load_biotrans(model_path=model_path)
    embedding = bio_trans.compute_embeddings([seq])
    embed = pd.DataFrame(embedding['full'][0])
    return embed


def build_position_data(seq, features_dict, domain_ranges, include_embedding=True, model_path=None):
    """
    build dataset on a sequence position basis
    don't put the label in features_dict, use build_label to handle label

    :param domain_ranges: dict of key = domain name, value = domain range for domain features
    :type domain_ranges: dict
    :param model_path: if model path is provided, load pretrained biotransformer for embedding
    :type model_path: str
    :param include_embedding: if true, get the embedding vector features from biotransformer
    :type include_embedding: bool
    :param seq: sequence of interest
    :type seq: str
    :param features_dict: dictionary of key = feature name, value = position dictionary (key = pos, value=value)
    :type features_dict: dict
    :return: position dataset
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(list(range(0, len(seq))))
    df.columns = ['pos']
    for feat_name, pos_dict in features_dict.items():
        df[feat_name] = df['pos'].map(pos_dict)
    df = df.set_index('pos')
    for domain_name, domain_idx in domain_ranges.items():
        df[[domain_name + '_before', domain_name, domain_name + '_after']] = df.apply(
            lambda x: domain_positions(x, domain_idx), axis=1, result_type='expand')
    if include_embedding:
        embed = position_embedding_features(seq, model_path)
        df = df.join(embed)
    return df


def build_label(pos_df, pos_muts, N_mut, label_name='tp'):
    """
    for the position dataframe, set the label ('TP') to 1 if # muts at position is >= N_mut, 0 otherwise
    in the pos_df, the index indicates position
    :param pos_df: position dataframe dataset
    :type pos_df: pandas.dataframe
    :param pos_muts: dict of key=position, value = calculated n muts
    :type pos_muts: dict
    :param N_mut: value of mutations required to be considered "mutable" (TP = 1)
    :type N_mut: int
    :return: position dataframe with TP label
    :rtype: pandas.DataFrame
    """
    pos_df['pos_muts'] = pos_df.index.map(pos_muts)
    pos_df[label_name] = -1
    pos_df.loc[pos_df['pos_muts'] < N_mut, label_name] = 0
    pos_df.loc[pos_df['pos_muts'] >= N_mut, label_name] = 1
    pos_df = pos_df.drop('pos_muts', axis='columns')
    return pos_df


def mutable_label(row, pos_muts, N_mut):
    """
    function to create a 1 (mutable)/0 (conserved) label with a dictionary that holds #mutations for each position

    :param row: position data row, the index is position number
    :type row: pandas.Series
    :param pos_muts: dict of key = position, value = calculated n muts
    :type pos_muts: dict
    :param N_mut: value of mutations required to be considered "mutable" (TP = 1)
    :type N_mut: int
    :return: 1 if mutable, 0 otherwise (conserved)
    :rtype: int
    """
    pos = row.name
    muts = pos_muts[pos]
    if muts >= N_mut:
        return 1
    else:
        return 0


def emerging_label(row, precut_label, postcut_label):
    """
    creates a 1 (emerging)/0 (non emerging) label from precut and postcut
    if precut is conserved and postcut is mutable, emerging = 1, 0 otherwise

    :param row: position data row, the index is position number
    :type row: pandas.Series
    :param precut_label: column name of precut mutable label
    :type precut_label: str
    :param postcut_label: column name of postcut mutable label
    :type postcut_label: str
    :return: 1 if emerging, 0 otherwise
    :rtype: int
    """
    precut = row[precut_label]
    postcut = row[postcut_label]
    if postcut == 1 and precut == 0:
        return 1
    else:
        return 0


def build_feature_dict(df, hyphy_data_path, pos_map=None, hyphy_groups=None):
    """
    builds the features dict needed for building position data model

    :param pos_map: map of sequence position mapped to reference position, if provided, then re-map hyphy position dicts
    :type pos_map: dictionary of key = reference positon, value = sequence position, deletions are "-"
    :param df: sequence probability dataframe
    :type df: pandas.DataFrame
    :param hyphy_data_path: path to the hyphy results
    :type hyphy_data_path: str
    :param hyphy_groups: groups to use in hyphy features, defaults to FEL and MEME
    :type hyphy_groups: list
    :return: dictionary of key = feature, value = feature dict to map to position df
    :rtype: dict
    """
    features_dict = {
        'all_entropy': nlargest_prob_entropy(df, 20),
        'top10_entropy': nlargest_prob_entropy(df, 10),
        'top5_entropy': nlargest_prob_entropy(df, 5),
        'top2_entropy': nlargest_prob_entropy(df, 2),
        'wt_prob': wt_prob_from_df(df)
    }
    top_n_feat = top_proba_feat(df)
    for k, v in top_n_feat.items():
        features_dict[k] = v
    if hyphy_groups is None:
        hyphy_groups = ['FEL', 'MEME']
    with open(hyphy_data_path, 'rb') as f:
        hyphy_data = pickle.load(f)
    hyphy_feat = hyphy_pipeline_features(hyphy_data, hyphy_groups)
    if pos_map is not None:
        for feat_name, values in hyphy_feat.items():
            values_remapped = remap_to_seq(pos_map, values)
            features_dict[feat_name] = values_remapped
    else:
        for feat_name, values in hyphy_feat.items():
            features_dict[feat_name] = values
    return features_dict


def domain_positions(row, domain_idx):
    """
    returns 1 hot encoding for position being before a domain, inside a domain, after a domain.

    :param row: row of dataframe for the position. The index is position index
    :type row: pandas.Series
    :param domain_idx: list of [start, stop] range of the domain with index starting at 1
    :type domain_idx: list
    :return: before, current, after one-hot encoded for the given domain
    :rtype: tuple
    """
    pos = row.name
    start = domain_idx[0] - 1
    stop = domain_idx[1] - 1
    if pos < start:
        before = 1
        current = 0
        after = 0
    elif start <= pos <= stop:
        before = 0
        current = 1
        after = 0
    else:
        before = 0
        current = 0
        after = 1
    return before, current, after


if __name__ == '__main__':
    ref_seq = load_ref_spike()
    ref_seq = str(ref_seq.seq)


    model_folder = 'logs/protbert_full_ft'
    model_path = last_biotrans_ft_path(model_folder)
    seq_prob_path = 'data/processed/ncbi_tree_v0/NSCS/tree_v0_seq_masked_prob_ft.pkl'
    with open(seq_prob_path, 'rb') as f:
        seq_proba = pickle.load(f)
    seq_proba = seq_proba[ref_seq]

    data = proba_to_df(ref_seq, seq_proba)
    hyphy_path = 'data/processed/hyphy/tree_v3_train/s_annot.pkl'
    feat_dict = build_feature_dict(data, hyphy_path)

    domain_ranges = {
        'bCoV_S1_N': [33, 337],
        'bCoV_S1_RBD': [348, 526],
        'CoV_S1_C': [536, 592],
        'CoV_S2': [711, 1232],
        'CoV_S2_C': [1233, 1272]
    }

    # TODO: allow for loading saved embedding values instead of having to re-calculate embedding
    data = build_position_data(ref_seq, feat_dict, domain_ranges, include_embedding=True, model_path=model_path)


    with open('data/processed/position_stats/pos_muts_pssm_postcutoff.pkl', 'rb') as a:
        postcut_muts = pickle.load(a)

    with open('data/processed/position_stats/pos_muts_pssm_precutoff.pkl', 'rb') as f:
        precut_muts = pickle.load(f)

    data['postcut_tp'] = data.apply(lambda x: mutable_label(x, postcut_muts, 5), axis=1)
    data['precut_tp'] = data.apply(lambda x: mutable_label(x, precut_muts, 5), axis=1)

    data['emerging_tp'] = data.apply(lambda x: emerging_label(x, 'precut_tp', 'postcut_tp'), axis=1)

    data.to_pickle('data/processed/conservation/conservationdata.pkl')

