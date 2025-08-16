import numpy as np
import scanpy as sc
import igraph as ig
import leidenalg

import random

from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, fcluster
from kneed import KneeLocator

from GTra import *
from preproc import *
from joblib import Parallel, delayed


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

# Cell type labeling
def add_annotation(obj, time):
    adata = obj.tp_data_dict[time]
    meta = adata.obs

    cname = meta.columns[0]
    obj.params.cell_type_label = cname

    # Filtering low-qualited celltypes
    cell_cnts = meta[cname].value_counts() > obj.params.filter_cell_n
    cts = [c for c, val in cell_cnts.to_dict().items() if val]

    adata = adata[meta[meta[cname].isin(cts)].index].copy()

    # Annotation
    clist = adata.obs[cname].values.tolist()
    ct_dict = {ct: c for c, ct in enumerate(np.unique(clist))}
    clabel = [int(ct_dict[ct]) for ct in clist]

    adata.obs["cluster_label"] = clabel

    K = max(clabel) + 1
    cells = adata.obs_names.tolist()
    label_index = get_label(clabel, cells, K)

    return adata, label_index, K, clabel

# Graph-based clustering (leiden)
def cell_graph_clustering(obj, time):
    adata = obj.tp_data_dict[time]

    # Dimension reduction and clustering
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=obj.params.cn_neighbors)
    sc.tl.leiden(adata, resolution=obj.params.cn_cluster_resolution)

    clabel = list(map(int, adata.obs.leiden.tolist()))

    adata.obs["cluster_label"] = clabel

    K = max(clabel) + 1
    cells = list(adata.obs.index)
    label_index = get_label(clabel, cells, K)

    return adata, label_index, K, clabel

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

# kNN-based gene clustering pipeline
def knn_based_gene_clustering(tp_data_df, tp_data_obs, clabel):
    cname = "cluster_label"

    cell_mask = (tp_data_obs[cname]==clabel).values
    counts = tp_data_df.loc[cell_mask,:].T

    # Normalization
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    log_expr = np.log1p(cpm)
    X = pd.DataFrame(zscore(log_expr, axis=1),
                     index=log_expr.index, columns=log_expr.columns)
    X = X.dropna()

    # Dimension reduction
    metric = "cosine"
    nn_num = min(20, X.shape[1] - 1)
    npcs = min(30, X.shape[1])

    X_pca = PCA(n_components=npcs).fit_transform(X)
    nn = NearestNeighbors(n_neighbors=nn_num, metric=metric).fit(X_pca)
    knn_graph = nn.kneighbors_graph(X_pca, mode="connectivity")
    src, tar = knn_graph.nonzero()
    g = ig.Graph(edges = list(zip(src, tar)), directed=False)

    # Resolution sweep
    res_range = np.arange(0.1, 1.5, 0.1)
    mod, par = [], []
    for res in res_range:
        part = leidenalg.find_partition(g,
                                        leidenalg.RBConfigurationVertexPartition,
                                        seed=1234,
                                        resolution_parameter=res
                                        )
        mod.append(part.modularity)
        par.append(part)
    
    # 5. Knee-point detection
    kneedle = KneeLocator(res_range, mod, curve="concave", direction="increasing")
    if kneedle.knee is None:
        best_idx = int(np.argmax(mod))
    else:
        best_idx = np.where(res_range == kneedle.knee)[0][0]

    # delta_mod = np.diff(mod)
    # second_jump_idx = np.argsort(delta_mod)[::-1][1]
    # best_part = par[second_jump_idx+1]
    best_part = par[best_idx]
    best_labels = best_part.membership

    num_clusters = len(set(best_labels))
    if num_clusters == 1:
        for part in reversed(par):
            if len(set(part.membership)) >= 2:
                best_part = part
                best_labels = best_part.membership
                break
    

    gnames = list(counts.index)
    glabel_idx = get_label(best_labels, gnames, max(best_labels)+1)
    return glabel_idx

# Extract required data of object
def extract_dat(obj):
    dat = {
        'tp_data_dict':obj.tp_data_dict,
        'tp_data_num':obj.tp_data_num,
        'genes':obj.genes,
        'params':obj.params
    }
    return dat

# Build boot obj from safe data
def build_boot_obj(dat):
    obj = GTraObject()
    obj.tp_data_dict = dat['tp_data_dict']
    obj.tp_data_num = dat['tp_data_num']
    obj.genes = dat['genes']
    obj.params = dat['params']
    return obj

# Selected randomized cells
def create_random_cells(obj, tp):
    trial_cells = list()
    dat = obj.tp_data_dict[tp]
    dat_group = dat.obs.groupby(obj.params.cell_type_label)
    for _, mat in dat_group:
        n_cells = list(mat.index)
        N = int(len(n_cells)*.8)
        trial_cells.extend(random.sample(n_cells, N))
    return trial_cells


# Parallel Clustering for statistical testing
def parallel_clustering(dat, idx, N):
    boot_obj = build_boot_obj(dat)

    # GTra's clustering
    for tp in range(boot_obj.tp_data_num):
        # Select random cells
        trial_cells = create_random_cells(boot_obj, tp)
        if len(trial_cells) == 0:
            print(f"[Warning] No cells selected for time {tp}, skipping.", flush=True)
            continue
        boot_obj.tp_data_dict[tp] = boot_obj.tp_data_dict[tp][trial_cells]

        # Cell clustering
        cell_clustering(boot_obj, tp)
        K = boot_obj.cell_optimal_k[tp]
        print(f"[Iter {idx+1}/{N}] Time {tp+1}/{boot_obj.tp_data_num} n_celltypes={K}", flush=True)

        n_jobs = 8
        arg_list = []
        for clabel in range(K):
            tp_data_df = boot_obj.tp_data_dict[tp].to_df()
            tp_data_obs = boot_obj.tp_data_dict[tp].obs

            args = (tp_data_df,tp_data_obs,clabel)
            arg_list.append(args)
        
        # Gene clustering
        glabels = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(knn_based_gene_clustering)(*args) for args in arg_list
            )
        boot_obj.gene_label_info[tp] = glabels
    
    # Edge selection
    boot_obj.cell_type_info = concat_meta(boot_obj)
    for t in range(boot_obj.tp_data_num-1):
        boot_obj.cal_edge_score(t, t+1)
        boot_obj.edge_rank_test(t, t+1)
        for i in range(len(boot_obj.selected_edges[t])):
            boot_obj.node_info.loc[boot_obj.node_cnt] = boot_obj.selected_edges[t][i]
            boot_obj.node_cnt += 1

    res = save_stat_res(boot_obj)
    gcinfo = get_gcinfo(boot_obj)

    # Remove memory allocation
    del boot_obj

    return res, gcinfo

# GTra's statistical testing
def parallel_testing(obj, N=10):
    from preproc import get_rankinfo, get_ccmatrix, rank_distribution, cal_pvals
    n_cores = 40
    # Perform N times
    dat = extract_dat(obj)
    res = Parallel(n_jobs=n_cores, backend='loky')(
        delayed(parallel_clustering)(dat, i, N) for i in range(N)
    )
    obj.ccmatrix = get_ccmatrix(res)

    merged_dat, merged_rank = get_rankinfo(res)

    # Calculating probability values
    cnt_dict = {}
    for key, subdict in merged_dat.items():
        new_dict = {}
        for subkey, value in subdict.items():
            new_dict[subkey] = (sum(value)/N)*100
        cnt_dict[key] = new_dict

    obj.cnt_dict = cnt_dict
    obj.rank_dict = merged_rank

    # Selecting candidate edges with probability values exceeding a certain threshold
    candidate_dict = {}
    for inter, subdict in cnt_dict.items():
        candidate_pair = []
        for key, val in subdict.items():
            if val < obj.params.static_th: continue
            candidate_pair.append(key)
        candidate_dict[inter] = candidate_pair 
        
    # Calculates p-value
    obj.dist_df = rank_distribution(obj)        
    obj.pval_df = cal_pvals(obj.dist_df)

    obj.candidate_dict = candidate_dict # candidate edges between time intervals
    obj.static_flag = True

# Get consensus-cluster results
def get_cc_clusters(obj):
    from gutils import calc_gap

    cc_dict = obj.ccmatrix.copy()
    genes = obj.genes.copy()

    for time in range(obj.tp_data_num):
        clabel_clusters = []
        for clabel in range(obj.cell_optimal_k[time]):
            cc = cc_dict[time][clabel]
            linked = linkage(cc, "ward")
            K = calc_gap(linked)
            clusters = fcluster(linked, K, criterion="maxclust")

            clustered_genes = []
            for cl in range(1,max(clusters)+1):
                tmp = [gene for gene, label in zip(genes, clusters) if label==cl]
                clustered_genes.append(tmp)
            clabel_clusters.append(clustered_genes)
        
        obj.gene_label_info[time] = clabel_clusters
    
    







        

