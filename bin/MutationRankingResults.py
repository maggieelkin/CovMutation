import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np
import pandas as pd
import seaborn as sns
import colorcet as cc
from matplotlib.legend_handler import HandlerTuple
from matplotlib import pyplot


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


def mark_significant(seq_muts, sig_mut_lst):
    """
    function to take seq_mut dict (key = mutated sequence, value = meta data with probability and mutation string)
    and re-mark what mutations are significant
    :param seq_muts: dict (key = mutated sequence, value = meta data with probability and mutation string)
    :type seq_muts: dict
    :param sig_mut_lst: list of significant mutations formatted as [wt][pos+1][mut]
    :type sig_mut_lst: lst
    :return: dictionary with recalculated significant in meta
    :rtype: dict
    """
    for seq, meta in seq_muts.items():
        meta['significant'] = meta['mutation'] in sig_mut_lst
    return seq_muts


def results_over_thresholds(seq_mutations, sig_muts, seq_identifier, mapped_cnt, mut_map=None, parent_in_train=None,
                            **ranking_values):
    """

    :param seq_mutations:
    :type seq_mutations:
    :param sig_muts:
    :type sig_muts:
    :param seq_identifier:
    :type seq_identifier:
    :param mapped_cnt:
    :type mapped_cnt:
    :param mut_map:
    :type mut_map:
    :param parent_in_train:
    :type parent_in_train:
    :return:
    :rtype:
    """
    result_list = []
    if mut_map is not None:
        ref_muts = [mut_map[x] for x in sig_muts]
    else:
        ref_muts = sig_muts
    result_meta = {}
    if seq_identifier != 'ref_seq':
        result_meta['parent_hash'] = seq_identifier
    else:
        result_meta['parent_hash'] = 'ref'
    if parent_in_train is not None:
        result_meta['parent_in_train'] = parent_in_train

    result_meta.update({
        'result_type': 'Combined',
        'threshold': np.nan,
        'muts': "; ".join(sig_muts),
        'ref_muts': '; '.join(ref_muts),
        'n_gt': len(ref_muts)
    })
    results = seq_mutation_data_results(seq_mutations, **ranking_values)
    result_meta.update(results)
    result_list.append(result_meta.copy())
    for sig_mut in sig_muts:
        new_sig_muts = [sig_mut]
        if mut_map is not None:
            ref_mut = mut_map[sig_mut]
        else:
            ref_mut = sig_mut
        freq = mapped_cnt[sig_mut]
        result_meta['result_type'] = 'Solo Mutation'
        result_meta['threshold'] = freq
        result_meta['ref_muts'] = ref_mut
        result_meta['muts'] = sig_mut
        result_meta['n_gt'] = 1
        seq_mutations = mark_significant(seq_mutations, new_sig_muts)
        results = seq_mutation_data_results(seq_mutations, **ranking_values)
        result_meta.update(results)
        result_list.append(result_meta.copy())
    return result_list


def prep_results_for_graph(results_summary):
    rd = pd.melt(results_summary, id_vars=['threshold', 'n_muts'],
                 value_vars=['cscs_auc', 'change_auc', 'prob_auc', 'dnms_auc', 'attn_auc'])
    rd = rd.rename(columns={'n_muts': "Number Mutations"})
    rd['variable'] = rd['variable'].replace({'dnms_auc': 'DNMS',
                                             'cscs_auc': 'CSCS',
                                             'change_auc': 'Semantic Change',
                                             'prob_auc': "Grammaticality",
                                             'attn_auc': 'Attention Change'})
    return rd


def plot_thresholds(results_summary, save_path=None):
    rd = prep_results_for_graph(results_summary)
    n = len(set(rd['variable']))
    palette = sns.color_palette(cc.glasbey, n_colors=n + 1)
    palette.pop(4)
    custom_params = {"axes.spines.right": False, "axes.spines.top": False, 'axes.edgecolor': 'black'}
    sns.set_theme(style="ticks", rc=custom_params, font_scale=1.5, font='Dejavu Sans')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    hue_order = ['DNMS', 'CSCS', 'Semantic Change', 'Grammaticality', 'Attention Change']
    sns.lineplot(data=rd, x='threshold', y='value', hue='variable', hue_order=hue_order,
                 palette=palette, linewidth=2.0, ax=ax)

    sns.scatterplot(data=rd, x='threshold', y='value', hue='variable', legend='brief',
                    size='Number Mutations', sizes=(10, 200), palette=palette, ax=ax, hue_order=hue_order)

    handles, labels = ax.get_legend_handles_labels()

    labels_names = labels
    leg1 = pyplot.legend([tuple([handles[0], handles[6]]),
                          tuple([handles[1], handles[7]]),
                          tuple([handles[2], handles[8]]),
                          tuple([handles[3], handles[9]]),
                          tuple([handles[4], handles[10]])],
                         labels_names[:5], handlelength=2.0, markerscale=1.5,
                         handler_map={tuple: HandlerTuple(ndivide=1)},
                         loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    for line in leg1.get_lines():
        line.set_linewidth(2.5)

    leg2 = pyplot.legend([handles[12], handles[13], handles[14], handles[15]],
                         labels_names[12:], handler_map={tuple: HandlerTuple(ndivide=None)},
                         title='Number Mutations', loc="upper left",
                         bbox_to_anchor=(1, .5), frameon=False, ncol=1)
    leg2._legend_box.align = "left"
    pyplot.gca().add_artist(leg1)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color="grey", alpha=0.25, linestyle="--", zorder=0)
    ax.set_xlabel('Mutation Frequency Threshold')
    plt.xscale('log')
    plt.ylabel('AUC')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
