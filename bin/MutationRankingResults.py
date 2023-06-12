import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd


# ################################ OLD METHODS TO CALCULATE AUC ##################################### #
def cscs(df, gt_column, prob_col='prob', change_col='change', plot=False, flip_change_rank=True, plot_title=None,
         save_path=None):
    """
    function to return CSCS AUC from a dataframe of mutations.
    dataframe will need to include the mutation (each row per mutation), probability and semantic change
    ground truth column (bool field) will show which ones are counted in top N correct
    this is adapted from Learning the Language of Viral Evolution and Escape

    :param save_path:
    :type save_path:
    :param change_col: column in df that has the semantic change values, default is "change"
    :type change_col: str
    :param prob_col: column in df that has the grammaticality values, default is "prob"
    :type prob_col: str
    :param df: dataframe of probability, semantic change and GT value for each mutation
    :type df: pandas.DataFrame
    :param gt_column: ground truth column name (bool field in dataframe)
    :type gt_column: string
    :param plot: if true, show and return a plot of AUC
    :type plot: bool
    :param flip_change_rank: default is smallest to largest
    :type flip_change_rank: bool
    :param plot_title: title for plot
    :type plot_title: str
    :return: CSCS AUC,  Change AUC, prob auc
    :rtype: bool
    """
    prob = np.array(df[prob_col])
    change = np.array(df[change_col])
    df[gt_column] = df[gt_column].astype(bool)
    gt_idx = np.array(df[gt_column])
    # number of ground truth to search for (Y Axis)
    n_gt = sum(gt_idx)
    # number of mutations tested (X Axis)
    max_consider = len(prob)
    # X axis
    n_consider = np.array([i + 1 for i in range(max_consider)])
    # norm AUC
    norm = max_consider * n_gt
    beta = 1
    # CSCS combines change and prob
    if flip_change_rank:
        acquisition = ss.rankdata(-change) + (beta * ss.rankdata(prob))
    else:
        # Original Science Paper method
        acquisition = ss.rankdata(change) + (beta * ss.rankdata(prob))
    cscs_rank = ss.rankdata(-acquisition)[gt_idx]
    n_cscs = np.array([sum(cscs_rank <= i + 1) for i in range(max_consider)])
    cscs_auc = auc(n_consider, n_cscs) / norm
    # prob
    prob_rank = ss.rankdata(-prob)[gt_idx]
    n_prob = np.array([sum(prob_rank <= i + 1) for i in range(max_consider)])
    prob_auc = auc(n_consider, n_prob) / norm
    # change
    if flip_change_rank:
        change_rank = ss.rankdata(change)[gt_idx]
    else:
        # original science paper method
        change_rank = ss.rankdata(-change)[gt_idx]
    n_change = np.array([sum(change_rank <= i + 1) for i in range(max_consider)])
    change_auc = auc(n_consider, n_change) / norm
    if plot:
        plt.figure()
        plt.plot(n_consider, n_cscs)
        plt.plot(n_consider, n_change, c='C0', linestyle='-.')
        plt.plot(n_consider, n_prob, c='C0', linestyle=':')
        plt.plot(n_consider, n_consider * (n_gt / max_consider),
                 c='gray', linestyle='--')

        plt.xlabel(r'$ \log_{10}() $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')

        plt.legend([
            r'$ \Delta \mathbf{\hat{z}} + ' +
            r'\beta \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $,' +
            (' AUC = {:.3f}'.format(cscs_auc)),
            r'$  \Delta \mathbf{\hat{z}} $ only,' +
            (' AUC = {:.3f}'.format(change_auc)),
            r'$ \hat{p}(x_i | \mathbf{x}_{[N] ∖ \{i\} }) $ only,' +
            (' AUC = {:.3f}'.format(prob_auc)),
            'Random guessing, AUC = 0.500'
        ])
        plt.xlabel('Top N')
        plt.ylabel('Number of escape mutations in top N')
        if plot_title:
            plt.title(plot_title)
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)
    return cscs_auc, change_auc, prob_auc


def ranking_auc(value_array, gt_idx, rank_high_low=True):
    """

    :param rank_high_low:
    :type rank_high_low:
    :param value_array:
    :type value_array:
    :param gt_idx:
    :type gt_idx:
    :return:
    :rtype:
    """
    n_gt = sum(gt_idx)
    # number of mutations tested (X Axis)
    max_consider = len(value_array)
    # X axis
    n_consider = np.array([i + 1 for i in range(max_consider)])
    # norm AUC
    norm = max_consider * n_gt
    if rank_high_low:
        val_rank = ss.rankdata(-value_array)[gt_idx]
    else:
        val_rank = ss.rankdata(value_array)[gt_idx]
    n_ranked = np.array([sum(val_rank <= i + 1) for i in range(max_consider)])
    ranked_auc = auc(n_consider, n_ranked) / norm
    mean_rank = np.mean(val_rank)
    max_rank = np.max(val_rank)
    min_rank = np.min(val_rank)
    results = {'auc': ranked_auc, 'mean_rank': mean_rank, 'max_rank': max_rank, 'min_rank': min_rank,
               'n_ranked': n_ranked, 'n_consider': n_consider, 'n_gt': n_gt, 'max_consider': max_consider,
               'random_yaxis': n_consider * (n_gt / max_consider)}
    return results


def attn_cscs_old(df, gt_column, prob_col='prob', change_col='change', attn_col='attn', alpha=1, beta=1, gamma=1):
    """

    :param df:
    :type df:
    :param gt_column:
    :type gt_column:
    :param prob_col:
    :type prob_col:
    :param change_col:
    :type change_col:
    :param attn_col:
    :type attn_col:
    :return:
    :rtype:
    """
    prob = np.array(df[prob_col])
    change = np.array(df[change_col])
    attn = np.array(df[attn_col])
    df[gt_column] = df[gt_column].astype(bool)
    gt_idx = np.array(df[gt_column])

    acquisition = (alpha * ss.rankdata(-change)) + (beta * ss.rankdata(prob))
    cscs_auc = ranking_auc(acquisition, gt_idx)
    cscs_auc = cscs_auc['auc']

    new_acquisition = (alpha * ss.rankdata(-change)) + (beta * ss.rankdata(prob)) + (gamma * ss.rankdata(-attn))
    n_cscs = ranking_auc(new_acquisition, gt_idx)
    attn_cscs_auc = n_cscs['auc']

    prob = ranking_auc(prob, gt_idx)
    prob_auc = prob['auc']

    # semantic change
    change = ranking_auc(change, gt_idx, rank_high_low=False)
    change_auc = change['auc']

    # Attn
    attn = ranking_auc(attn, gt_idx, rank_high_low=False)
    attn_auc = attn['auc']

    results = {'attn_cscs_auc': attn_cscs_auc,
               'cscs_auc': cscs_auc,
               'change_auc': change_auc,
               'prob_auc': prob_auc,
               'attn_auc': attn_auc}

    return results


# ################################################################################################ #

def attn_cscs(df, gt_column, prob_col='prob', change_col='change', attn_col='attn', change_rank_high_low=False,
              alpha=1, beta=1, gamma=1, plot=False):
    """

    :param df:
    :type df:
    :param gt_column:
    :type gt_column:
    :param prob_col:
    :type prob_col:
    :param change_col:
    :type change_col:
    :param attn_col:
    :type attn_col:
    :param alpha:
    :type alpha:
    :param beta:
    :type beta:
    :param gamma:
    :type gamma:
    :param plot:
    :type plot:
    :return:
    :rtype:
    """
    df['tp'] = 0
    df.loc[df[gt_column], 'tp'] = 1
    tp = np.array(df['tp'])

    prob = np.array(df[prob_col])
    # flip ranking for roc_curve
    prob_rank = ss.rankdata(prob)
    scores = {'prob_auc': prob_rank}

    if change_col in df:
        change = np.array(df[change_col])
        # flip ranking for roc_curve
        if change_rank_high_low is True:
            change_rank = ss.rankdata(change)
        else:
            change_rank = ss.rankdata(-change)
        scores['change_auc'] = change_rank
    else:
        change = None
    if attn_col in df:
        attn = np.array(df[attn_col])
        # flip ranking for roc_curve
        attn_rank = ss.rankdata(-attn)
        scores['attn_auc'] = attn_rank
    else:
        attn = None

    if change is not None:
        if change_rank_high_low is True:
            acquisition = (alpha * ss.rankdata(change)) + (beta * ss.rankdata(prob))
        else:
            acquisition = (alpha * ss.rankdata(-change)) + (beta * ss.rankdata(prob))
        acq_rank = ss.rankdata(acquisition)
        scores['cscs_auc'] = acq_rank
    if change is not None and attn is not None:
        #TODO: Rename method with new name
        new_acquisition = (alpha * ss.rankdata(-change)) + (beta * ss.rankdata(prob)) + (gamma * ss.rankdata(-attn))
        new_acq_rank = ss.rankdata(new_acquisition)
        scores['attn_cscs_auc'] = new_acq_rank

    results = {}
    for k, v in scores.items():
        fpr, tpr, thresholds = roc_curve(tp, v)
        AUC = auc(fpr, tpr)
        results[k] = AUC
        if plot:
            lab = "{0}, AUC:{1}".format(k, str(round(AUC, 4)))
            plt.plot(fpr, tpr, label=lab)
    if plot:
        plt.plot([0, 1], [0, 1], c='gray', linestyle='--', label='Random guessing, AUC = 0.500')
        plt.legend()
        plt.title('AUC calculated with ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    return results


def seq_mutation_data_results(seq_data, **kwargs):
    """
    takes sequence mutation dictionary (from get_seq_mutation_dict) and returns AUC
    can include semantic change or exclude
    requires probability

    :param seq_data: dictionary of key = mut_str, value = meta
    :type seq_data: dict, pd.DataFrame
    :return: dictionary of auc results
    :rtype: dict
    """
    if isinstance(seq_data, dict):
        df = pd.DataFrame(seq_data.values())
    else:
        df = seq_data
    results = attn_cscs(df, 'significant', **kwargs)
    if len(kwargs) > 0:
        results.update(kwargs)
    else:
        defaults = {'alpha': 1, 'beta': 1, 'gamma':1}
        results.update(defaults)
    return results
