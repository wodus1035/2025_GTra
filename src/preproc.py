import numpy as np
import pandas as pd
import copy
import os

from scipy.stats import mannwhitneyu
from collections import Counter, defaultdict
from cluster_func import *


def filter_genes(adata):
    genes = []
    dat = adata.to_df().T
    for g, val in dat.iterrows():
        zero_ratio = list(val).count(0) / len(val)
        if zero_ratio > .95: continue
        genes.append(g)
    return genes

# Merge cell type information total time points
def concat_meta(obj):
    merge_cell_type_df = {}
    for time in range(obj.tp_data_num):
        cell_type_df = obj.tp_data_dict[time].obs[[obj.params.cell_type_label]]
        if time == 0:
            merge_cell_type_df = cell_type_df
        else:
            merge_cell_type_df = pd.concat([merge_cell_type_df, cell_type_df])
    return merge_cell_type_df

# Edge statistic testing
def save_stat_res(obj):
    est_edges = {}
    rank_edges = {}
    for _, v in obj.node_info.iterrows():
        # Get node name
        tok_s = v[0].split('_')
        tok_t = v[1].split('_')
        # Get label info
        t1, ct1 = int(tok_s[0][1:]),int(tok_s[1])
        ct2 = int(tok_t[1])
  
        # Save edge info
        edge_name = f'{ct1}_{ct2}'

        # Initialization
        if est_edges.get(t1) is None:
            est_edges[t1] = {}

        if rank_edges.get(t1) is None:
            rank_edges[t1] = {}

        if est_edges[t1].get(edge_name) is None:
            est_edges[t1][edge_name] = 1

        if rank_edges[t1].get(edge_name) is None:
            rank_edges[t1][edge_name] = v[-1]

    return est_edges, rank_edges

# Get gene cluster's info
def get_gcinfo(obj):
    gene_list = obj.genes
    tp_gcinfo = dict()
    for tp in range(obj.tp_data_num):
        n_celltypes = obj.cell_optimal_k[tp]
        ct_gclusters = []
        for clabel in range(n_celltypes):
            n_gc = obj.gene_label_info[tp][clabel]
            gene_to_cluster = {gene: gc for gc, gs in enumerate(n_gc) for gene in gs}
            gclusters = [gene_to_cluster.get(g, 0) for g in gene_list]
            ct_gclusters.append(gclusters)
            
        tp_gcinfo[tp] = ct_gclusters
    return tp_gcinfo

# Combining parallel processing results
def get_rankinfo(res):
    merged_dat = defaultdict(lambda: defaultdict(list))
    merged_rank = defaultdict(lambda: defaultdict(list))
    for dat in res:
        for key, subdict in dat[0][0].items():
            for subkey, value in subdict.items():
                merged_dat[key][subkey].extend([value])

        for key, subdict in dat[0][1].items():
            for subkey, value in subdict.items():
                merged_rank[key][subkey].extend([value])

    merged_dat = {key: dict(subdict) for key, subdict in merged_dat.items()}
    merged_rank = {key: dict(subdict) for key, subdict in merged_rank.items()}

    return merged_dat, merged_rank

# Get parallel results
def get_ccmatrix(res):
    ccmatrix = defaultdict(dict)
    N = len(res)
    
    for time in res[0][1].keys():
        cts = len(res[0][1][time])
        for ct in range(cts):
            labels = res[0][1][time][ct]
            n_genes = len(labels)
            mat = np.zeros((n_genes, n_genes))
            
            for run in res:
                labels = np.array(run[1][time][ct])
                same_cluster = (labels[:, None] == labels[None, :]).astype(int)
                mat += same_cluster 
            mat /= N
            ccmatrix[time][ct] = mat
    return ccmatrix

# Store rank dstribution information
def rank_distribution(obj):
    ct_label_dict = dict()
    for tp in range(obj.tp_data_num):
        cell_clustering(obj, tp)
        id = obj.tp_data_dict[tp].obs.value_counts().index
        ct_label = {str(i[1]): i[0] for i in id}
        ct_label_dict[tp] = ct_label

    x = copy.deepcopy(obj.rank_dict)
    intervals = x.keys()
    iter_static = []
    for it in intervals:
        for st, vals in x[it].items():
            tok = st.split("_")
            source = ct_label_dict[it][tok[0]]
            target = ct_label_dict[it + 1][tok[1]]
            for v in vals:
                iter_static.append([it, source, target, v])

    dist_df = pd.DataFrame(
        iter_static, columns=["Interval", "source", "target", "rank_score"]
    )
    # fname = obj.params.answer_path_dir

    # if os.path.isfile(fname):
    #     obj.answer_path = pd.read_csv(fname, sep=",")
    #     answer_edges = set(zip(obj.answer_path["source"],obj.answer_path["target"]))
    #     filtered_df = dist_df[dist_df.apply(
    #         lambda row: (row["source"], row["target"]) in answer_edges, axis=1)]
    #     dist_df = filtered_df.copy()

    return dist_df
    
# p-value
def cal_pvals(dist_df):
    pval_list = []
    for it,df in dist_df.groupby('Interval'):
        for ct, cdf in df.groupby('source'):
            x = cdf.sort_values('rank_score')
            for tar in np.unique(x['target'].values):
                tar_score = cdf[cdf['target']==tar]['rank_score']
                remain_score = cdf[cdf['target']!=tar]['rank_score']
                
                if len(tar_score) > 0 and len(remain_score) > 0:
                    stat, pval = mannwhitneyu(tar_score, remain_score, alternative='less')
                    pval_list.append([it, ct, tar, pval])
                else:
                    # 로그 출력 또는 NaN 처리
                    print(f"[Skip] Empty group at Interval={it}, source={ct}, target={tar}")
                    pval_list.append([it, ct, tar, np.nan])
                
                # stat, pval = mannwhitneyu(tar_score, remain_score, alternative='less')
                # pval_list.append([it, ct, tar, pval])

    pval_df = pd.DataFrame(pval_list, columns=['Interval','source','target','p-value'])
    return pval_df