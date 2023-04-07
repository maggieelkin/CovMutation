import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd


def mut_freq_auc(df, mut_freqs, prob_col, plot):
    """
    given a dataframe with change and prob values, return AUC for mutations with frequency within the range mut_freqs

    :param plot: if true, plot the results
    :type plot: bool
    :param prob_col: probability column header in df
    :type prob_col: str
    :param df: model results with n_times marking the frequency of the mutation in the dataset
    :type df: pandas.DataFrame
    :param mut_freqs: list of upper bound frequencies for mutations
    :type mut_freqs: list
    :return: result dataframe
    :rtype: pd.DataFrame
    """
    results_df = pd.DataFrame()
    index = 0
    for n_mut in mut_freqs:
        df['tp'] = 0
        df.loc[df['n_times'] > n_mut, 'tp'] = 1
        if plot:
            print('N mutation threshold: {}'.format(n_mut))
        results = calculate_rank_auc(df, prob_col=prob_col, plot=plot)
        results['n_mut_thres'] = n_mut
        results_row = pd.DataFrame(results, index=[index])
        results_df = results_df.append(results_row)
        index = index + 1
    return results_df


def calculate_rank_auc(df, tp_column='tp', prob_col='prob', plot=False):
    """
    different method of calculating CSCS AUC
    uses roc_curve, so needs to flip ranking than used in cscs function
    typically goes faster than the cscs function

    :param df:
    :type df:
    :param tp_column: column name from df that specifies 1 if mutation is significant, 0 otherwise
    :type tp_column: str
    :param prob_col:
    :type prob_col:
    :param plot:
    :type plot:
    :return:
    :rtype:
    """
    df['prob_rank'] = ss.rankdata(df[prob_col])
    df['change_rank'] = ss.rankdata(-df['change'])
    df['acq_rank'] = ss.rankdata(df['change_rank'] + df['prob_rank'])
    num_tp = np.sum(df[tp_column])
    list_score = ['change_rank', 'acq_rank', 'prob_rank']
    results = {'num_tp': num_tp}
    for score in list_score:
        fpr, tpr, thresholds = roc_curve(df[tp_column].values, df[score].values)
        AUC = auc(fpr, tpr)
        results[score] = AUC
        if plot:
            lab = "{0}_AUC:{1}".format(score, str(round(AUC, 2)))
            plt.plot(fpr, tpr, label=lab)
    if plot:
        plt.legend()
        plt.title('N Mutations {}'.format(num_tp))
        plt.show();
    return results


def cscs(df, gt_column, prob_col='prob', change_col='change', plot=False, flip_change_rank=True, plot_title=None,
         save_path=None):
    """
    function to return CSCS AUC from a dataframe of mutations.
    dataframe will need to include the mutation (each row per mutation), probability and semantic change
    ground truth column (bool field) will show which ones are counted in top N correct
    this is adapted from Learning the Language of Viral Evolution and Escape

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


def prob_auc_solo(df, gt_column, plot=False, plot_title=None, save_path=None):
    """
    function to return probability AUC only from a dataframe of mutations.
    dataframe will need to include the mutation (each row per mutation), probability and semantic change
    ground truth column (bool field) will show which ones are counted in top N correct
    this is adapted from Learning the Language of Viral Evolution and Escape
    :param df: dataframe of probability, semantic change and GT value for each mutation
    :type df: pandas.DataFrame
    :param gt_column: ground truth column name (bool field in dataframe)
    :type gt_column: string
    :param plot: if true, show and return a plot of AUC
    :type plot: bool
    :param plot_title: title for plot
    :type plot_title: str
    :return: CSCS AUC,  Change AUC, prob auc
    :rtype: bool
    """
    prob = np.array(df['prob'])
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
    # prob
    prob_rank = ss.rankdata(-prob)[gt_idx]
    n_prob = np.array([sum(prob_rank <= i + 1) for i in range(max_consider)])
    prob_auc = auc(n_consider, n_prob) / norm
    if plot:
        plt.figure()
        plt.plot(n_consider, n_prob)
        plt.plot(n_consider, n_consider * (n_gt / max_consider),
                 c='gray', linestyle='--')

        plt.xlabel(r'$ \log_{10}() $')
        plt.ylabel(r'$ \log_{10}(\Delta \mathbf{\hat{z}}) $')

        plt.legend([
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
    return prob_auc


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


def plot_rank_auc_results(x_axis, y_axis, plt_dict, plot_title=None, save_path=None):
    """

    :param save_path:
    :type save_path:
    :param plot_title:
    :type plot_title:
    :param x_axis:
    :type x_axis:
    :param y_axis:
    :type y_axis:
    :param plt_dict:
    :type plt_dict:
    :return:
    :rtype:
    """
    line_styles = ['-', '-.', ':']
    plt.figure()
    plt.plot(x_axis, y_axis,
             c='gray', linestyle='--', label='Random guessing, AUC = 0.500')
    i = 0
    for k, v in plt_dict.items():
        plt.plot(x_axis, v, linestyle=line_styles[i], label=k, c='C0')
        i = i + 1
    plt.xlabel('Top N')
    plt.ylabel('Number of escape mutations in top N')
    plt.legend()
    if plot_title:
        plt.title(plot_title)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def mutation_rank_results(array, gt_idx, title, plot=True, **kwargs):
    """

    :param title:
    :type title:
    :param array:
    :type array:
    :param gt_idx:
    :type gt_idx:
    :param plot:
    :type plot:
    :return:
    :rtype:
    """
    results = ranking_auc(array, gt_idx)
    if plot:
        legend_title = title + ", AUC = {}".format(round(results['auc'], 4))
        plt_dict = {legend_title: results['n_ranked']}
        plot_rank_auc_results(results['n_consider'], results['random_yaxis'], plt_dict, **kwargs)
    return results


def seq_mutation_dict_results(seq_mutation_dict):
    """
    takes sequence mutation dictionary (from get_seq_mutation_dict) and returns AUC
    can include semantic change or exclude
    requires probability
    :param seq_mutation_dict: dictionary of key = mut_str, value = meta
    :type seq_mutation_dict: dict
    :return: dictionary of auc results
    :rtype: dict
    """
    df = pd.DataFrame(seq_mutation_dict.values())
    results = {}
    if 'change' in df.columns.tolist():
        cscs_auc, change_auc, prob_auc = cscs(df, 'significant')
        results['cscs_auc'] = cscs_auc
        results['change_auc'] = change_auc
        results['prob_auc'] = prob_auc
    else:
        prob_auc = prob_auc_solo(df, 'significant')
        results['prob_auc'] = prob_auc
    return results
