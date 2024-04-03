import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd


def dnms(df, gt_column, prob_col='prob', change_col='change', attn_col='attn', change_rank_high_low=False,
         alpha=1, beta=1, gamma=1, plot=False):
    """

    :param change_rank_high_low:
    :type change_rank_high_low:
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
        new_acquisition = (alpha * ss.rankdata(-change)) + (beta * ss.rankdata(prob)) + (gamma * ss.rankdata(-attn))
        new_acq_rank = ss.rankdata(new_acquisition)
        scores['dnms_auc'] = new_acq_rank

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
    results = dnms(df, 'significant', **kwargs)
    if len(kwargs) > 0:
        results.update(kwargs)
    else:
        defaults = dict(alpha=1, beta=1, gamma=1)
        results.update(defaults)
    return results
