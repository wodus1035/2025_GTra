import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg
import igraph as ig
import copy

import random

import seaborn as sns
import matplotlib.colors as mcolors

from scipy.stats import zscore


from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster

from kneed import KneeLocator

from joblib import Parallel, delayed

from .core import *
from .preproc import *

## Get clustering results
def get_label(label,names,K):
    label_index = []
    for k in range(K):
        tmp_idx = []
        for cidx in range(len(label)):
            if label[cidx] == k:
                tmp_idx.append(names[cidx])
        label_index.append(tmp_idx)
    return label_index

# Get label index
def _get_label(labels, names, K):
    label_index = [[] for _ in range(K)]
    for name, lbl in zip(names, labels):
        label_index[lbl].append(name)
    return label_index


## --------------- Cell clustering --------------- ##
# Cell type labeling
def add_annotation(obj, time):
    adata = obj.tp_data_dict[time]
    adata.layers["raw"] = adata.X.copy()

    cname = adata.obs.columns[0]
    obj.params.cell_type_label = cname

    # 1) Filter out low-count cell types
    counts = adata.obs[cname].value_counts()
    valid_ct = counts[counts > obj.params.filter_cell_n].index
    adata = adata[adata.obs[cname].isin(valid_ct)].copy()

    # 2) Convert cell types into integer labels
    unique_ct = sorted(valid_ct)
    ct2id = {ct: i for i, ct in enumerate(unique_ct)}
    clabel = adata.obs[cname].map(ct2id).astype(int).tolist()

    adata.obs["cluster_label"] = clabel
    K = len(unique_ct)
    
    # 3) Build label index list for each cluster
    cells = adata.obs_names.tolist()
    label_index = _get_label(clabel, cells, K)

    return adata, label_index, K, clabel


# Graph-based cell clustering using leiden
def cell_graph_clustering(obj, time):
    adata = obj.tp_data_dict[time]
    if "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()

    # 1) Normalize -> log -> scale
    if "norm" not in adata.layers:
        adata.layers["norm"] = adata.X.copy()
        sc.pp.normalize_total(adata, layer="norm")
        sc.pp.log1p(adata, layer="norm")
        sc.pp.scale(adata, max_value=10, layer="norm")

    # 2) PCA + neighbors + Leiden
    adata.X = adata.layers["norm"]
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=obj.params.cn_neighbors,
                    use_rep="X_pca")
    sc.tl.leiden(adata, resolution=obj.params.cn_cluster_resolution)
    
    # 3) Cluster labeling
    clabel = adata.obs["leiden"].astype(int)
    adata.obs["cluster_label"] = clabel

    K = adata.obs["cluster_label"].nunique()
    cells = adata.obs_names.tolist()
    label_index = _get_label(clabel, cells, K)

    return adata, label_index, K, clabel


def create_color_dict(obj):
    all_cts = []
    ct_label = obj.params.cell_type_label
    for _, dat in obj.tp_data_dict.items():
        all_cts.extend(dat.obs[ct_label].unique())
    
    unique_cts = sorted(set(all_cts))
    palette = sns.color_palette('Set2',len(unique_cts))
    celltype_colors = dict(zip(unique_cts, palette))
    obj.celltype_colors = celltype_colors


# Cell clustering for each time point
def cell_clustering(obj, time):
    if obj.params.label_flag:
        adata, cli, K, clabel = add_annotation(obj, time)
    else:
        adata, cli, K, clabel = cell_graph_clustering(obj, time)
    
    obj.tp_data_dict[time] = adata
    obj.cell_optimal_k[time] = K
    obj.cell_cluster_label[time] = clabel
    obj.cell_label_index[time] = cli


## --------------- Gene clustering --------------- ##
# kNN based gene clustering
# def knn_based_gene_clustering(X, obs, target_cluster, gene_names=None):
#     """
#     Perform fast KNN-based gene clustering for a single cell cluster.
#     Takes genes x cells expression, builds a cosine-KNN graph, 
#     runs Leiden clustering, and returns gene groups.
#     """
#     cname = "cluster_label"

#     # --- 1) Select cells of the target cluster --- #
#     mask = obs[cname].values == target_cluster
#     if mask.sum() <= 2:
#         return []
    
#     X_sub = X[mask]  # shape = (#cells, #genes)
#     # convert to dense if sparse
#     if not isinstance(X_sub, np.ndarray):
#         X_sub = X_sub.toarray()
    
#     # gene x cell
#     G = X_sub.T

#     # ---- 2) Normalize: CPM + log1p + zscore ---- #
#     # # 2-1) CPM (safe version)
#     # rsum = G.sum(axis=1, keepdims=True)               # (genes, 1)
#     # zero_mask = (rsum == 0)                           # True for genes with zero total expression

#     # rsum_safe = np.where(zero_mask, 1, rsum)          # avoid division by zero

#     # G_cpm = (G / rsum_safe) * 1e6                     # CPM transform
#     # G_cpm[zero_mask[:, 0]] = 0                        # rows that had zero expr remain zero
    
#     # log1p
#     G_log = np.log1p(G)
#     # z-score normalization across cells (axis=0)
#     # G_norm = zscore(G_log, axis=1, nan_policy='omit')
#     # Replace remaining NaN with 0
#     G_norm = np.nan_to_num(G_log)
#     # Now gene × cell convert
#     G_final = G_norm  # shape = (#genes, #cells)

#     # ---- 3) SVD embedding of genes ---- #
#     n_cells = G_final.shape[1]
#     n_components = min(20, n_cells - 1)
#     svd = TruncatedSVD(n_components=n_components, random_state=0)
#     G_svd = svd.fit_transform(G_final)

#     # ---- 4) Build KNN graph ---- #
#     nn_num = min(20, n_cells - 1)
#     nn = NearestNeighbors(n_neighbors=nn_num, metric="cosine").fit(G_svd)
#     knn_graph = nn.kneighbors_graph(G_svd, mode="connectivity")
#     src, tar = knn_graph.nonzero()

#     g = ig.Graph(n=G_final.shape[0], edges=list(zip(src, tar)), directed=False)

#     # ---- 5) Leiden clustering ---- #
#     res_list = [0.2, 0.4, 0.6, 0.8]

#     best_mod = -999
#     best_part = None

#     for res in res_list:
#         part = leidenalg.find_partition(
#             g,
#             leidenalg.RBConfigurationVertexPartition,
#             resolution_parameter=res,
#             seed=1234
#         )
#         if part.modularity > best_mod:
#             best_mod = part.modularity
#             best_part = part

#     if best_part is None:
#         return []

#     labels = best_part.membership

#     # Force split if only 1 cluster
#     if len(set(labels)) == 1:
#         part = leidenalg.find_partition(
#             g,
#             leidenalg.RBConfigurationVertexPartition,
#             resolution_parameter=1.6,
#             seed=1234
#         )
#         labels = part.membership

#     # ---- 6) Convert into glabel_idx format ---- #
#     K = max(labels) + 1
#     glabel_idx = [[] for _ in range(K)]

#     for gi, lbl in enumerate(labels):
#         glabel_idx[lbl].append(gene_names[gi] if gene_names is not None else gi)
#     return glabel_idx
def knn_based_gene_clustering(
    tp_data_df,
    tp_data_obs,
    clabel,
    gnames,
    max_pcs=30,
    max_knn=20,
    random_state=1234
):
    """
    v1.2: paper-safe & optimized gene clustering
    """

    cname = "cluster_label"

    # --------------------------------------------------
    # Select cells
    # --------------------------------------------------
    cell_mask = (tp_data_obs[cname]==clabel).values
    counts = tp_data_df.loc[cell_mask,:].T

    # Normalization
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_expr = np.log1p(cpm)
    X = pd.DataFrame(zscore(log_expr, axis=1),
                     index=log_expr.index, columns=log_expr.columns)
    X = X.dropna()

    # --------------------------------------------------
    # PCA (numpy only)
    # --------------------------------------------------
    n_cells = X.shape[1]
    npcs = min(max_pcs, n_cells - 1)
    X_pca = PCA(n_components=npcs, random_state=random_state).fit_transform(X)

    # --------------------------------------------------
    # kNN graph (cosine, weighted)
    # --------------------------------------------------
    nn_num = min(max_knn, n_cells - 1)
    nn = NearestNeighbors(
        n_neighbors=nn_num,
        metric="cosine"
    ).fit(X_pca)

    knn_dist = nn.kneighbors_graph(
        X_pca, mode="distance"
    )
    src, tar = knn_dist.nonzero()
    weights = 1.0 - knn_dist.data  # cosine similarity

    g = ig.Graph(
        edges=list(zip(src, tar)),
        edge_attrs={"weight": weights},
        directed=False
    )

    # --------------------------------------------------
    # Resolution sweep (restricted, stable)
    # --------------------------------------------------
    res_range = np.arange(0.2, 1.3, 0.1)
    modularities, partitions = [], []

    for res in res_range:
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            seed=random_state,
            resolution_parameter=res
        )
        modularities.append(part.modularity)
        partitions.append(part)

    # Knee-point detection
    kneedle = KneeLocator(
        res_range,
        modularities,
        curve="concave",
        direction="increasing"
    )

    if kneedle.knee is not None:
        best_idx = np.where(res_range == kneedle.knee)[0][0]
    else:
        best_idx = int(np.argmax(modularities))

    best_part = partitions[best_idx]
    labels = best_part.membership

    # --------------------------------------------------
    # Fallback: enforce >= 2 clusters
    # --------------------------------------------------
    if len(set(labels)) < 2:
        for part in reversed(partitions):
            if len(set(part.membership)) >= 2:
                labels = part.membership
                break

    # --------------------------------------------------
    # Output
    # --------------------------------------------------
    glabel_idx = get_label(labels, gnames, max(labels) + 1)
    return glabel_idx


## --------------- bootstrapping based clustering --------------- ##


# Create object for bootstrapping
def _extract_dat(obj):
    return {
        'tp_data_dict':obj.tp_data_dict,
        'tp_data_num':obj.tp_data_num,
        'genes':obj.genes,
        'params':obj.params
    }

# Build mini GTra's object
def _build_boot_obj(dat):
    from .core import GTraObject
    obj = GTraObject() # type: ignore
    obj.tp_data_dict = dat["tp_data_dict"]
    obj.tp_data_num = dat["tp_data_num"]
    obj.genes = dat["genes"]
    obj.params = dat["params"]
    return obj

# Select randomized cell type-specific cells
def _create_random_cells(obj, time):
    trial_cells = []
    dat = obj.tp_data_dict[time]
    dat_group = dat.obs.groupby(obj.params.cell_type_label)
    for _, mat in dat_group:
        n_cells = list(mat.index)
        N = int(len(n_cells)*.8)
        trial_cells.extend(random.sample(n_cells, N))
    return trial_cells


# Perform gene clustering for each time point
def process_timepoint(dat, tp):
    """
    Process a single time point for GTra bootstrapping.

    Steps:
        1. Build a temporary bootstrap object.
        2. Randomly subsample cells for the target time point.
        3. Perform cell-level clustering (Leiden or annotation-based).
        4. Extract the subset expression matrix and metadata.
        5. Perform gene-level clustering for each discovered cell cluster.

    Returns:
        A dictionary containing:
            - tp: time point index
            - K: number of detected cell clusters
            - clabel: cell cluster label vector
            - cli: list of cell indices per cluster
            - glabels: list of gene-cluster assignments per cell cluster
    """
    mini = _build_boot_obj(dat)
    
    # 1) Random cell selection
    trial_cells = _create_random_cells(mini, tp)
    if len(trial_cells) == 0:
        return dict(tp=tp, K=0, clabel=None, cli=None, glabels=None)
    
    # Slice AnnData safetly
    adata = mini.tp_data_dict[tp]
    mini.tp_data_dict[tp] = adata[adata.obs_names.isin(trial_cells)].copy()

    # 2) Cell clustering
    cell_clustering(mini, tp)
    K = mini.cell_optimal_k[tp]
    clabel = mini.cell_cluster_label[tp]
    cli = mini.cell_label_index[tp]

    # 3) Prepare numeric matrix
    X = mini.tp_data_dict[tp].to_df()
    obs = mini.tp_data_dict[tp].obs
    gene_names = mini.tp_data_dict[tp].var_names.tolist()

    # 4) Gene clustering
    glabels = []
    for cid in range(K):
        gcl = knn_based_gene_clustering(X, obs, cid, gene_names)
        glabels.append(gcl)
    
    return dict(
        tp=tp,
        K=K,
        clabel = clabel,
        cli=cli,
        glabels=glabels
    )

# Perform Step 1 and 2
def Run_step1_and_2(dat):
    boot_obj = _build_boot_obj(dat)
    T = boot_obj.tp_data_num
    
    # Stage 1: timepoint-level sequential
    results = []
    for tp in range(T):
        res = process_timepoint(dat, tp)
        results.append(res)
    # Store cluster results to object
    for res_tp in results:
        tp = res_tp["tp"]
        boot_obj.gene_label_info[tp] = res_tp["glabels"]
        boot_obj.cell_cluster_label[tp] = res_tp["clabel"]
        boot_obj.cell_label_index[tp] = res_tp["cli"]
        boot_obj.cell_optimal_k[tp] = res_tp["K"]
    
    # Stage 2: Edge score & rank test
    boot_obj.cell_type_info = concat_meta(boot_obj)
    all_edges = []
    for t in range(T-1):
        boot_obj.cal_edge_score(t, t+1)
        boot_obj.edge_rank_test(t, t+1)
        
        all_edges.extend(boot_obj.selected_edges[t])
    
    boot_obj.node_info = pd.DataFrame(all_edges)
    boot_obj.node_cnt = len(all_edges)
    
    res = save_stat_res(boot_obj)
    gcinfo = get_gcinfo(boot_obj)
    
    del boot_obj
    
    return res, gcinfo

# Store score distribution information
def _score_distribution(obj):
    """
    obj.score_dict: {interval: {'s_t': [scores]}}
    boj.tp_data_dict[tp].obs: cluster_label -> cell types
    """
    
    ct_label_dict = dict()
    for tp in range(obj.tp_data_num):
        cell_clustering(obj, tp)
        id = obj.tp_data_dict[tp].obs.value_counts().index
        ct_label = {str(i[1]): i[0] for i in id}
        ct_label_dict[tp] = ct_label
    
    x = copy.deepcopy(obj.score_dict)
    intervals = x.keys()
    iter_static = []
    for it in intervals:
        for st, vals in x[it].items():
            tok = st.split('_')
            source = ct_label_dict[it][tok[0]]
            target = ct_label_dict[it+1][tok[1]]
            
            for v in vals:
                iter_static.append([it, source, target, v])
    dist_df = pd.DataFrame(
        iter_static, columns=["Interval", "source","target", "score"]
    )
    return dist_df
    

def statistical_testing(obj, N=50, n_cores=8):
    """
    Completely optimized statistical testing pipeline.
    - Bootstrap iterations run outer loop
    - Each iteration internally parallel over timepoints
    - Merging vectorized
    - Candidate edge extraction accelerated
    """
    
    dat =_extract_dat(obj)
    
    # 1) Run N bootstrap iterations
    res = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(Run_step1_and_2)(dat)
        for _ in range(N)
        )
    
    # 2) Build CCM (correspondence matrix)
    obj.ccmatrix = get_ccmatrix(res)
    
    # 3) Merge edge information (score + occurrence)
    merged_dat, merged_score = get_scoreinfo(res)
    obj.score_dict = merged_score
    
    # 4) Compute probability (% occurrence)
    cnt_dict = {}
    for inter, subdict in merged_dat.items():
        cnt_dict[inter] = {edge: (sum(vals)/N)*100
                                  for edge, vals in subdict.items()}
        obj.cnt_dict = cnt_dict
    
    # 5) Select candidate edges by threshold
    th = obj.params.static_th
    candidate_dict = {}
    for inter, subdict in cnt_dict.items():
        cand = [edge for edge, v in subdict.items() if v >= th]
        candidate_dict[inter] = cand
    
    obj.candidate_dict = candidate_dict
    obj.static_flag = True
    
    # 6) Compute score distribution
    obj.dist_df = _score_distribution(obj)
    
    # 7) Compute p-values
    obj.pval_df = cal_pvals(obj.dist_df)
    

def _calc_gap(linked, min_k=2, max_k=None):
    """
    Select optimal K by largest distance jump in hc
    """
    d = linked[:, 2] # linkage distance
    delta_d = np.diff(d)
    jump_idx = np.argmax(delta_d)
    
    # linkage result has n-1 merges
    n = linked.shape[0] + 1
    optimal_k = n - jump_idx
    
    if max_k is None:
        max_k = n // 2
    optimal_k = max(min_k, min(optimal_k, max_k))
    
    return optimal_k


def cc_clustering(obj):
    """
    Generate gene modules for each time point and each cell cluster.

    For every (time point, cell cluster), this function:
      - reads the consensus gene–gene similarity matrix (obj.ccmatrix)
      - performs hierarchical clustering (Ward linkage)
      - determines optimal module number (via GAP heuristic)
      - assigns genes to modules using fcluster

    Results are stored in:
        obj.gene_label_info[time] = list of gene modules per cell cluster.
    """
    cc_dict = obj.ccmatrix.copy()
    genes = obj.genes.copy()
    
    for time in range(obj.tp_data_num):
        clabel_clusters = []
        for clabel in range(obj.cell_optimal_k[time]):
            cc = cc_dict[time][clabel]
            linked = linkage(cc, "ward")
            K = _calc_gap(linked)
            clusters = fcluster(linked, K, criterion="maxclust")
            
            clustered_genes = []
            for cl in range(1, max(clusters)+1):
                tmp = [gene for gene, label in zip(genes, clusters) if label==cl]
                clustered_genes.append(tmp)
            clabel_clusters.append(clustered_genes)
        
        obj.gene_label_info[time] = clabel_clusters
                
    
    