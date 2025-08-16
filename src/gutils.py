import numpy as np
import pandas as pd
import scipy.stats as ss

import random
import math

import plotly.graph_objects as go

import networkx as nx
import os
import re

from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix

from scipy.interpolate import CubicSpline

## -------------------------- Functions for step 2 --------------------------##
def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def normalize(v):
    vmag=magnitude(v)
    return [v[i]/vmag for i in range(len(v))]

def vect_mu(v_list):
    v = np.array(v_list)
    vsum=np.ndarray.sum(v,axis=0)
    return normalize(vsum)

# Calculate jaccard similarity
def JS(l1, l2):
    a = set(l1).union(set(l2))
    b = set(l1).intersection(set(l2))

    return len(b) / len(a), b

# Min max scaling
def min_max_scaling(l1, l2):
    mm_model = MinMaxScaler()
    l1 = mm_model.fit_transform(np.reshape(np.array(l1),(-1,1)))
    l2 = mm_model.fit_transform(np.reshape(np.array(l2),(-1,1)))
    return l1.flatten(), l2.flatten()

# Update index
def update_idx(t1,t2,t1_s,t1_g,t2_s,t2_g):
    if t2_s == len(t2) - 1:
        if t2_g == len(t2[t2_s]) - 1:
            t2_s, t2_g = 0, 0
            t1_g += 1
            if t1_g == len(t1[t1_s]):
                t1_s += 1
                t1_g = 0
        else:
            t2_g += 1
    else:
        if t2_g < len(t2[t2_s]) - 1:
            t2_g += 1
        else:
            t2_s += 1
            t2_g = 0

    return t1_s, t1_g, t2_s, t2_g

# Calculate edge connection number
def cal_connection(js_list):
    # Calculate # of cell to cell connection
    ctc = {}
    for i in range(len(js_list)):
        if ctc.get(tuple([js_list[i][0],js_list[i][2]])) is None:
            ctc[tuple([js_list[i][0], js_list[i][2]])] = 1
        else:
            ctc[tuple([js_list[i][0], js_list[i][2]])] += 1
    ctc_n = len(ctc)

    # Calculate # of gene to gene connection
    gtg_n = 0
    for k,v in ctc.items():
        gtg_n += v
    
    return ctc_n, gtg_n

# Jaccard similarity distribution
def JS_distribution(obj, tp1, tp2):
    # Load gene sets of each time point
    t1 = obj.gene_label_info[tp1]
    t2 = obj.gene_label_info[tp2]

    # Sample and gene cluster index (init)
    t1_s, t1_g, t2_s, t2_g = 0, 0, 0, 0

    # Comparison intersected genes between adjacent time points
    js_list = []
    while (len(t1) != t1_s) and (len(t1[t1_s]) != t1_g):
        # JS analysis
        sim, inter_genes = JS(t1[t1_s][t1_g], t2[t2_s][t2_g])

        # If size of gene set == 0 then continue
        if len(inter_genes) == 0:
            t1_s,t1_g,t2_s,t2_g = update_idx(t1,t2,t1_s,t1_g,t2_s,t2_g)
            continue
        
        # Save JS results
        js_list.append([t1_s,t1_g,t2_s,t2_g,sim])
        t1_s, t1_g, t2_s, t2_g = update_idx(t1,t2,t1_s,t1_g,t2_s,t2_g)
    
    return js_list

# Threshold test
def JS_threshold_test(obj, tp1, tp2):
    SIM_PARAM = 4
    sort_js = sorted(JS_distribution(obj, tp1, tp2), key=lambda x:x[SIM_PARAM], reverse=True)

    # Calculate # of connected edges 
    th_list, ctc_list, gtg_list = [], [], []
    threshold = 0.0

    for _ in range(100):
        new_js = []
        for s in sort_js:
            if s[SIM_PARAM] > threshold:
                new_js.append(s)
        threshold += 0.01
        ctc_n, gtg_n = cal_connection(new_js)

        th_list.append(threshold)
        ctc_list.append(ctc_n)
        gtg_list.append(gtg_n)

        if ctc_n == 0 or gtg_n == 0:
            break

    # Min max scaling
    ctc_scale, gtg_scale = min_max_scaling(ctc_list, gtg_list)
    obj.ctc_list[tp1] = ctc_scale
    obj.gtg_list[tp1] = gtg_scale

    # Calculate max gap
    gap_ctg = []
    for c,g in zip(ctc_scale, gtg_scale):
        gap_ctg.append(np.abs(c-g))
    
    # Select optimal threshold
    sorted_gap_ctg = sorted(gap_ctg, reverse=True)
    optimal_th = th_list[gap_ctg.index(sorted_gap_ctg[1])]

    return optimal_th

# Create normalized pseudo-bulk data
def cal_cos_dist(obj,tp1,tp2,t1_s,t2_s,t1_df,t2_df,inter_genes):
    # Load cells
    t1_cells = obj.cell_label_index[tp1][t1_s]
    t2_cells = obj.cell_label_index[tp2][t2_s]

    # Calculate average of data
    t1_pseudo = t1_df.loc[list(inter_genes),t1_cells].apply(pd.to_numeric).mean(axis=1)
    t2_pseudo = t2_df.loc[list(inter_genes),t2_cells].apply(pd.to_numeric).mean(axis=1)

    # Check not expressed genes
    pseudo_concat = pd.concat([t1_pseudo,t2_pseudo],axis=1)
    expr_gene_idx = [i for i,val in enumerate(pseudo_concat.sum(axis=1)) if val]
    expr_genes = list(pseudo_concat.iloc[expr_gene_idx,:].index)

    # Calculate cosine distance
    if len(expr_genes) == 0:
        return -1
    else:
        pseudo_dat = pd.concat([t1_pseudo,t2_pseudo],axis=1).to_numpy()
        # Calculate distance
        centroid = vect_mu(pseudo_dat)

        # Check dat
        def check_cosine(u, v):
            if np.allclose(u, 0) or np.allclose(v, 0):
                return 1
            else:
                return distance.cosine(u,v)
        # cos_dists = [distance.cosine(v,centroid) for v in pseudo_dat]
        cos_dists = [check_cosine(v,centroid) for v in pseudo_dat]
        
    return cos_dists

# Load edge info
def get_edge_info(obj, tp1):
    # Sorting candidate edges using cosine distance
    COS_PARAM = 5
    sort_edge_info = sorted(obj.edge_info[tp1], key=lambda x:x[COS_PARAM])

    # Save edge value for cell type of tp1
    edge_info_dict = {}
    source_genes_info = obj.gene_label_info[tp1]
    for c in range(len(source_genes_info)):
        for edge in sort_edge_info:
            if edge[0] == c:
                if c not in edge_info_dict:
                    edge_info_dict[c] = [edge[1:]]
                else:
                    edge_info_dict[c].append(edge[1:])
    
    return edge_info_dict

# Calculate rank values of edges
def cal_rank(edge_weight, sw, dw):
    edge_weight = np.array(edge_weight)
    candidate_n = len(edge_weight)+1

    SIM_PARAM, COS_PARAM = 3, 5
    sim_rank = candidate_n - ss.rankdata(edge_weight[:,SIM_PARAM], method='min')
    dist_rank = ss.rankdata(edge_weight[:,COS_PARAM], method='min')
    rank_sum = ((sw*sim_rank)+(dw*dist_rank))/2

    # Save rank test results
    conversion = []
    for i in range(len(edge_weight)):
        origin = list(edge_weight[i])
        origin.extend([round(list(rank_sum)[i],3)])
        conversion.append(origin)

    return conversion

# Convert label name
def conv_label(label):
    time = int(label[label.find('t')+1:label.find('_')])
    cell = int(label[label.find('_')+1:label.rfind('_')])
    gene = int(label[label.rfind('_')+1:])
    return time, cell, gene

# Get inter genes
def get_inter_genes(obj, path):
    time, cell, gene = conv_label(path[0])
    inter_genes = set(obj.gene_label_info[time][cell][gene])

    for i in range(1, len(path)):
        time, cell, gene = conv_label(path[i])
        gene_set = set(obj.gene_label_info[time][cell][gene])
        inter_genes = inter_genes.intersection(gene_set)

    return inter_genes

# Create candidate paths
def get_networks(obj, sub_graphs):
    # Get node info
    TIME_POS = 0
    nodes = []
    for sub in sub_graphs:
        node = list(sub.nodes())
        node.sort(key=lambda x: float(re.sub(r'\D','',x.split('_')[TIME_POS])))
        nodes.append(node)

    # Create path 
    networks = []
    inter_gene_th = 10
    for i in range(len(nodes)):
        node_name = nodes[i]
        sources = [t for t in node_name if node_name[0][:node_name[0].find('_')] in t]
        targets = [t for t in node_name if node_name[-1][:node_name[-1].find('_')] in t]

        # Save path lists
        candidate_path = []
        for source in sources:
            for target in targets:
                sub_paths = list(nx.all_simple_paths(sub_graphs[i],source=source,target=target))
                for path in sub_paths:
                    inter_genes = get_inter_genes(obj,path)
                    if len(inter_genes) > inter_gene_th:
                        candidate_path.append(path)
                        obj.path_gene_sets[tuple(path)] = inter_genes
        
        networks.extend(candidate_path)

    return networks

# Get unique cell type list
def get_unique_celltype(obj, tp, s):
    label = obj.cell_label_index[tp][s]
    name = obj.params.cell_type_label
    cts = dict(Counter(obj.cell_type_info.loc[label,name].values.tolist()))
    sorted_cts = sorted(cts.items(), key=lambda x:x[1], reverse=True)
    return sorted_cts[0][0]

# Check standard path
def check_standard_path(obj, tp1, t1_s, top_rank, einfo, einfo_results):
    # Paths that match the source cell type
    source_ct = get_unique_celltype(obj, tp1, t1_s)
    
    # Load standard path info
    fname = obj.params.answer_path_dir
    
    
    path_pvals = obj.pval_df.copy()
    sig_paths = path_pvals[(path_pvals["Interval"]==tp1)]
    sig_paths = sig_paths[(sig_paths["p-value"]<=obj.pval_th)]
    answers = sig_paths[sig_paths["source"]==source_ct]['target'].values.tolist()

    if os.path.isfile(fname):
        obj.answer_path = pd.read_csv(fname, sep=",")
        paths = obj.answer_path.copy()
        answer_input = sum(paths.loc[paths['source']==source_ct,
                                ['target']].values.tolist(),[])
        answers = list(set(answer_input).intersection(set(answers))).copy()
        
    
    # Filter incorrected paths among paths
    for r in range(top_rank):
        t2_s = int(einfo[r,:3][1])
        target_ct = get_unique_celltype(obj, tp1+1, t2_s)
        
        # If target cell type exist not in answer path then continue
        if target_ct not in answers: continue
        
        label_info = list(map(int,einfo[r,:3]))
        score_info = list(einfo[r,3:])
        einfo_results.append([t1_s]+label_info+score_info)
        
    return einfo_results

# compute gap between distances
# def calc_gap(linked):
#     d = linked[:,2]
#     delta_d = np.diff(d)
#     jump_idx = np.argsort(delta_d)[::-1]
#     second_idx = jump_idx[1]
#     optimal_k = len(d) - second_idx
#     return optimal_k


def calc_gap(linked, min_k=2, max_k=None):
    """
    Select optimal K by largest distance jump in hierarchical clustering.
    """
    d = linked[:, 2]  # linkage distances
    delta_d = np.diff(d)
    jump_idx = np.argmax(delta_d)

    # linkage result has n-1 merges → K = n - jump_idx
    n = linked.shape[0] + 1
    optimal_k = n - jump_idx

    # Boundaries
    if max_k is None:
        max_k = n // 2
    optimal_k = max(min_k, min(optimal_k, max_k))

    return optimal_k

## -------------------------- Functions for step 3 --------------------------##

# Constructing cell type-specific trajectory
def group_cell_type_trajectory(net_info):
    merge_path_dict = {}
    for path in net_info:
        cell_type_path = [node[:node.rfind('_')] for node in path]
        key = tuple(cell_type_path)

        if merge_path_dict.get(key) is None:
            merge_path_dict[key] = [path]
        else:
            merge_path_dict[key].append(path)

    return merge_path_dict

# Concat expression matrix
def merge_expr(obj, path):
    # Concat time-series expression matrix
    path_genes = list(obj.path_gene_sets[tuple(path)])
    expr_df, times = pd.DataFrame(), []
    for i, node in enumerate(path):
        tname = int(node[node.find('t')+1:node.find('_')])
        cname = int(node[node.find('_')+1:node.rfind('_')])
        
        cnames = list(obj.cell_label_index[tname][cname])
        expr = obj.tp_data_dict[tname].to_df().T.loc[path_genes,cnames]
        expr = expr.apply(pd.to_numeric).mean(axis=1)

        if i == 0:
            expr_df = expr
        else:
            expr_df = pd.concat([expr_df, expr],axis=1)
        
        times.append(f't{str(tname)}')

    # Check not expression genes
    no_zero = [i for i,val in expr_df.iterrows() if sum(val)]
    expr_df = expr_df.loc[no_zero,:]
    expr_df.columns = times
    expr_df.index.name = 'GeneID'

    return expr_df

# Select optimal K
def elbow_method(dat):
    distortion = []
    for k in range(2, 11):
        spk = SphericalKMeans(n_clusters=k,random_state=25,
                              max_iter=100).fit(csr_matrix(l2norm(dat)))
        distortion.append(spk.inertia_)
    
    gaps = list(abs(np.diff(distortion)))
    sorted_gaps = sorted(gaps,reverse=True)
    if len(gaps) == 1 or max(distortion) < 0.01:
        optimal_k = 2
    else:
        optimal_k = gaps.index(sorted_gaps[1])+3
    return optimal_k

# Calculate interval confidence
def cal_ic(df):
    n = len(df) # freedom
    std_err = np.std(df.to_numpy()) / n**0.5 # std error
    ic = ss.t.interval(0.95, n, list(df.mean().values.real), scale=std_err)
    return ic

# L2 normalization
def l2norm(dat):
    norm = np.sqrt(np.sum(np.square(dat), axis=1))
    norm = np.array(norm).reshape((-1, 1))
    norm = dat / norm
    return norm

## Filtering patterns
def pattern_filtering(obj):
    # Load key info
    pt_keys = list(obj.merge_pattern_dict.keys())
    
    # Filtering time-series gene expression patterns
    pattern_dict = {}
    variance_th = 0.2
    for key in pt_keys:
        # Get clustered patterns
        pt_df = l2norm(obj.merge_pattern_dict[key])
        centroid = list(pt_df.mean().values.real)
        
        if (len(pt_df.columns) != obj.tp_data_num):
            continue

        var_ic = np.max(abs(cal_ic(pt_df)[1]-centroid))

        if var_ic > variance_th:
            # print(f'check: {str(var_ic)}')
            continue

        pattern_dict[key] = obj.merge_pattern_dict[key]
    return pattern_dict

# Select candidate keys
def get_candidate_keys(keys):
    # Select pattern keys
    candidate_keys = {}
    for k in keys:
        key_group = k[:k.find('_')]
        if candidate_keys.get(key_group) is None:
            candidate_keys[key_group] = [k]
        else:
            candidate_keys[key_group].append(k)
    return candidate_keys

# Calculate pearson correlation
def cal_corr(mp_dict, ptc, pcut=0.05):
    candidate_pair = set()
    # Pattern comparison
    for i,s in enumerate(ptc):
        for j,t in enumerate(ptc):
            if i>=j: continue
            source = l2norm(mp_dict[s]).mean().values
            target = l2norm(mp_dict[t]).mean().values
            if ((np.isnan(source)) | (np.isnan(target))).any(): continue
            stat, pvals = ss.pearsonr(source, target)
            if (pvals>pcut) or (stat<=0): continue
            # Save candidate pairs
            pairs = tuple(sorted((s,t),key=lambda x:int(x.replace('_',''))))
            candidate_pair.add(pairs)
    
    return candidate_pair

# Merge candidate patterns
def select_patterns(ptc, mp_dict, sub_net, key, th=0.2):
    pt_dict = {}
    candidates = ptc.copy()
    for i,sub in enumerate(sub_net):
        for j,node in enumerate(sub):
            candidates.remove(node)
            if j == 0: m = mp_dict[node]
            else: m = pd.concat([m,mp_dict[node]])
        var_ic = np.max(cal_ic(l2norm(m))[1]-list(l2norm(m).mean().values))
        if var_ic > th: continue
        
        m = m.drop_duplicates()
        pt_dict[f'{key}_M{i}'] = m
    
    for sel_key in candidates:
        pt_dict[sel_key] = mp_dict[sel_key].copy()
    
    return pt_dict

# Merge similar patterns
def merge_sim_patterns(obj):
    mp_dict = obj.merge_pattern_dict.copy()
    candidate_keys = get_candidate_keys(mp_dict.keys())

    new_mp_dict = {}
    for key, ptc in candidate_keys.items():
        pairs = cal_corr(mp_dict, ptc)
        if len(pairs) == 0:
            for k in ptc:
                new_mp_dict[k] = mp_dict[k]
        else:
            pair_net = pd.DataFrame(pairs, columns=['s','t'])
            g = nx.from_pandas_edgelist(pair_net, 's','t', create_using=nx.DiGraph())
            sub_net = list(g.subgraph(c) for c in nx.weakly_connected_components(g))
            pt_dict = select_patterns(ptc, mp_dict, sub_net, key)
            new_mp_dict.update(pt_dict)
    
    return new_mp_dict

# Renaming pattern name
def renaming_pattern_id(mp_dict):
    unique_path = np.unique([i[:i.find('_')] for i in mp_dict.keys()])
    for path in unique_path:
        candidates = [k for k in mp_dict.keys() if k[:k.find('_')] == path]
        for i, key in enumerate(candidates):
            mp_dict[f'{path}_{i}'] = mp_dict.pop(key)
    # Sorting pattenr id
    mp_dict = dict(sorted(mp_dict.items(), key=lambda x:int(x[0][:x[0].find('_')])))
    return mp_dict

# Save pattern centroid
def save_pattern_centroid(obj):
    pt_csv_dict = {}
    for key, df in obj.merge_pattern_dict.items():
        mean_val = l2norm(df).mean().values
        pt_csv_dict[key] = mean_val
    
    pt_csv_df = pd.DataFrame(pt_csv_dict).T
    pt_csv_df.columns = ['T'+str(i+1) for i in range(len(pt_csv_df.columns))]
    pt_csv_df.index.name = 'Key'

    output_name = f'{obj.params.output_dir}/{obj.params.output_name}_pattern_centroid.csv'
    pt_csv_df.to_csv(output_name)

## Renaming cell-state trajectories
def convert_path_name(obj, key):
    key_label = key[:key.find('_')]
    merge_path_keys = list(obj.merge_path_dict.keys())

    # Convert path name to cell type name
    convert_name = ''
    for _, path in enumerate(merge_path_keys[int(key_label)]):
        cluster_label = int(path[path.find('_') + 1:])
        time_label = int(path[path.find('t') + 1: path.find('_')])
        cells = obj.cell_label_index[time_label][cluster_label]
        cname = obj.params.cell_type_label
        
        unique_cell_type = dict(Counter(obj.cell_type_info.loc[cells,cname].values.tolist()))
        cell_type = sorted(unique_cell_type.items(), key=lambda x: x[1], reverse=True)[0][0]
        convert_name += cell_type + '->'

    return convert_name[:convert_name.rfind('->')]

## Make gene set data frame for cell trajectory
def make_gene_set_frame(idx, gene_set, pt_df, key, convert_name):
    if idx == 0:
        gene_set = pd.DataFrame(list(pt_df.index), columns=[convert_name + '[' + key + ']'])
    else:
        tmp = pd.DataFrame(list(pt_df.index), columns=[convert_name + '[' + key + ']'])
        gene_set = pd.concat([gene_set, tmp], axis=1)
    return gene_set

# Smoothing the pattern
def spline_func(x, y):
    f = CubicSpline(x, y, bc_type='natural')
    x_new = np.linspace(0, len(x)-1, 100)
    y_new = f(x_new)
    return x_new, y_new

# Pattern interpolation based on a confidence interval
def interval_spline(x, y):
    x_new = np.linspace(0, len(x)-1, 100)

    pos_f = CubicSpline(x, y[0], bc_type='natural')
    neg_f = CubicSpline(x, y[1], bc_type='natural')
    
    pos_y = pos_f(x_new)
    neg_y = neg_f(x_new)
    return pos_y, neg_y

# Plotting gene expression patterns
def plotting_patterns(pt_df, key, start_cells, ax, pos):
    # Cubic spline function version
    import seaborn as sns
    colors = sns.color_palette('colorblind', n_colors=9)

    cubic_x = [i for i in range(len(pt_df.columns))]
    cubic_y = list(pt_df.mean().values)
    x_new, y_new = spline_func(cubic_x, cubic_y)

    # Calculation interval confidence    
    ic = cal_ic(pt_df)
    pos_y, neg_y = interval_spline(cubic_x, ic)

    row = pos // 3
    col = pos % 3

    # ax[row][col].plot(x_new, y_new, color='r', linewidth=0.8, linestyle='dashed')
    ax[row][col].plot(x_new, y_new, color=colors[row], linewidth=2)
    # ax[row][col].plot(cubic_x, cubic_y, 'ro')
    ax[row][col].scatter(cubic_x, cubic_y, color=colors[row], edgecolor='black', s=70, zorder=3)

    # ax[row][col].plot(x_new, pos_y, linestyle='dashed', color='gray', linewidth=0.8)
    # ax[row][col].plot(x_new, neg_y, linestyle='dashed', color='gray', linewidth=0.8)
    ax[row][col].fill_between(x_new, pos_y, neg_y, color=colors[row], alpha=0.15)

    ax[row][col].grid(False)


    ax[row][col].set_xlabel(f'Time points', size=10)
    ax[row][col].set_ylabel('Normalized gene expression', size=10)
    ax[row][col].set_xticks(cubic_x, labels=pt_df.columns, rotation=45)
    ax[row][col].set_title(f'Start:{start_cells}, CL: {key}\n (N={len(pt_df)})')

## Functions for plotting cell trajectories

# Create cell count dataframe
def create_cell_cnt(obj):
    cell_cnt_dict = {}
    for ct in np.unique(obj.cell_type_info.values.reshape(-1)):
        cell_cnt_dict[ct] = dict()
        for t in range(obj.tp_data_num):
            obs = obj.tp_data_dict[t].obs[obj.params.cell_type_label].value_counts()
            if obs.get(ct) is None:
                cnt = 0
            else:
                cnt = obs[ct]
            cell_cnt_dict[ct][t] = cnt
    
    cell_cnt_df = pd.DataFrame(cell_cnt_dict).T
    cell_cnt_df = cell_cnt_df[sorted(cell_cnt_df.columns)]
    cell_cnt_df.columns = [f't{str(t+1)}'for t in cell_cnt_df.columns]
    return cell_cnt_df

## Select colors of each celltypes

def generate_palette_dict(cell_types):
    palette_dict = {}
    for idx, cell_type in enumerate(cell_types):
        # 랜덤한 rgba 값 생성
        rgba = (random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.randint(0, 255) / 255, random.uniform(0.5, 1.0))
        rgba_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"
        palette_dict[cell_type] = rgba_str
    return palette_dict

# Convert node name
def get_ct_label(s, ct_label_dict):
    tp  = s[s.find('t')+1:s.find('_')]
    label = s[s.find('_')+1:]
    ct = ct_label_dict[int(tp)][label]
    return ct

# load edge info of cell trajectories
def get_celltype_edges(obj):
    celltype_info = np.unique(obj.cell_type_info.values.reshape(-1))
    # Celltype numbering of each time points, time points: int, cell type: string
    ct_label_dict = dict()
    for tp in range(obj.tp_data_num):
        id = obj.tp_data_dict[tp].obs.value_counts().index
        ct_label = {str(i[1]):i[0] for i in id}
        ct_label_dict[tp] = ct_label

    celltypes_sb = {} # starting point
    for ct in celltype_info:
        celltypes_sb[ct] = ct
        
    # load cell trajectory of each cell types
    celltypes_bs = dict()
    for path in list(obj.merge_path_dict.keys()):
        for ndx in range(len(path)-1):
            src = path[ndx]
            tar = path[ndx+1]
            
            sname = get_ct_label(src, ct_label_dict)
            tname = get_ct_label(tar, ct_label_dict)

            if celltypes_bs.get(sname) is None:
                celltypes_bs[sname] = [tname]
            else:
                celltypes_bs[sname].append(tname)

    for key, val in celltypes_bs.items():
        celltypes_bs[key] = list(np.unique(val))
    return celltype_info, celltypes_sb, celltypes_bs

class GTra(object):
    def __init__(self, sid, infile):
        df=pd.read_csv(infile, index_col=0)
        self.sid=sid    # sample id
        self.label=None # sample label
        self.tp=len(df.columns[0].split('->'))  # tp info
        self.celltypes=set()
        for i in df.columns:
            tok=i.split('[')[0].split('->')
            for j in tok: 
                self.celltypes.add(j)
        self.celltypes=sorted(self.celltypes)
        self.pattern_info={x.split('[')[1].split(']')[0]:x.split('[')[0] for x in df.columns} # pattern info - cell types across the trajectory
        self.pattern={x.split('[')[1].split(']')[0]:df[x].dropna().tolist() for x in df.columns}    # list of genes in pattern
        self.centroid={x:None for x in self.pattern.keys()} # centroid of pattern
        centroid_f='%s/%s_pattern_centroid.csv'%(str(infile).rsplit('/', 1)[0], self.sid)
        df=pd.read_csv(centroid_f, index_col=0)
        for patid in self.pattern.keys():
            self.centroid[patid]=df[df.index==patid].values[0]

        # sankey info
        self.sankey_dat=[]                              # input data to sankey plots
        self.adj_ct={x:{} for x in range(self.tp-1)}    # sankey info
        self.adj_gn={x:{} for x in range(self.tp-1)}    # sankey info
        self.adj_gset={x:{} for x in range(self.tp-1)}  # sankey info
        
# load gtra result's information
def get_gtra_res(obj, celltypes_sb):
    extname='_pattern_genes.csv'
    fpath = f'{obj.params.output_dir}/{obj.params.output_name}{extname}'

    # Load gtra's results
    sid = str(fpath).split('/')[-1].split(extname)[0]
    sdat = GTra(sid, fpath)

    # Get the trajectories info
    tr_keys = set()
    for patid in sdat.pattern.keys():
        trj = sdat.pattern_info[patid].split('->')
        gene_set = set(sdat.pattern[patid])
        for idx, _ in enumerate(trj[:-1]):
            key = '%s*%s'%(trj[idx], trj[idx+1])
            tr_keys.add(key)
            if key not in sdat.adj_ct[idx]:
                sdat.adj_ct[idx][key] = 0
                sdat.adj_gn[idx][key] = 0
                sdat.adj_gset[idx][key] = set()
            sdat.adj_ct[idx][key]+=1
            sdat.adj_gset[idx][key].update(gene_set)
            sdat.adj_gn[idx][key]=len(sdat.adj_gset[idx][key])

    tr_keys=sorted(tr_keys)

    # write time point transitinos 
    print('summarizing trajectories of GTra samples...')
    celltypes=set()
    sub_celltypes=set()
    trtp_celltypes=set()

    sdat.sankey_dat=[]
    for tp in range(sdat.tp-1):
        for key in tr_keys:
            if key not in sdat.adj_ct[tp]:
                continue                
            src_ct=key.split('*')[0]
            tar_ct=key.split('*')[1]
            
            celltypes.add(celltypes_sb[src_ct])
            celltypes.add(celltypes_sb[tar_ct])
            sub_celltypes.add(src_ct)
            sub_celltypes.add(tar_ct)
            trtp_celltypes.add('t%d_%s'%(tp, src_ct))
            trtp_celltypes.add('t%d_%s'%(tp+1, tar_ct))
            sdat.sankey_dat.append('%s,t%d_%s,t%d_%s,%d,%d,%s'%(sid, tp, key.split('*')[0], tp+1, 
                                                                key.split('*')[1], sdat.adj_ct[tp][key], len(sdat.adj_gset[tp][key]), 
                                                                '|'.join(sorted(sdat.adj_gset[tp][key]))))

    trtp_celltypes=sorted(trtp_celltypes)
    sub_celltypes=sorted(sub_celltypes)
    
    return sdat, trtp_celltypes, sub_celltypes

# check full path
def get_full_paths(sdat,celltypes_bs, target_ct):
    nodes = []
    for _, line in enumerate(sdat.sankey_dat):
        thisct=line.strip().split(',')[1].split('_')[1]
        
        if thisct not in celltypes_bs[target_ct]: continue
        #
        sid, src, tar, tcnt, gcnt, glist=line.strip().split(',')
        nodes.append([src,tar])

    graphs = nx.from_pandas_edgelist(pd.DataFrame(nodes,columns=['src','tar']), 'src','tar', create_using=nx.DiGraph())

    starts, ends = [], []
    for i in graphs.nodes():
        tok = i.split('_')
        t = tok[0][tok[0].find('t')+1:]
        if int(t) == 0:
            starts.append(i)
        elif int(t) == sdat.tp-1:
            ends.append(i)
    
    full_path_nodes = []
    for path in nx.all_simple_paths(graphs, source=starts[0], target=ends):
        full_path_nodes.extend(path)
        
    return list(np.unique(full_path_nodes))

# gtra's trajectories info
def get_plot_dat(sdat, trtp_celltypes, trtp_celltypes_d ,celltypes_bs, target_ct, color_palette):
    p_src, p_tar = [], []
    p_val_tcnt, p_val_gcnt = [], []
    ct_gcnt = {x:0 for x in trtp_celltypes}
    
    # Load gtra res info
    exist_ = get_full_paths(sdat, celltypes_bs, target_ct)
    
    for _, line in enumerate(sdat.sankey_dat):
        thisct=line.strip().split(',')[1].split('_')[1]
        
        if thisct not in celltypes_bs[target_ct]: continue
        #
        sid, src, tar, tcnt, gcnt, glist=line.strip().split(',')
        #
        if (src not in exist_ ) | (tar not in exist_): continue
    
        p_src.append(trtp_celltypes_d[src])
        p_tar.append(trtp_celltypes_d[tar])
        p_val_tcnt.append(tcnt)
        p_val_gcnt.append(gcnt)
        ct_gcnt[tar]+=int(gcnt)
    
    # archive sankey data
    sank_sid_d={'sid':sid, 'source':p_src, 'target':p_tar, 'tcnt': p_val_tcnt, 'gcnt':p_val_gcnt}
    #
    ctlabels=[]
    ctcolors=[]
    unique_ct = []
    pos_info= {}
    
    for ct in trtp_celltypes:
        ctname = ct.split('_')[1]
        gcnt = ct_gcnt[ct]
    
        if gcnt:
            ctlabels.append('%s (%d)'%(ctname, gcnt))
            unique_ct.append(ctname)
            pos_info[f'{ctname} ({gcnt})'] = ct
        else:
            ctlabels.append('%s'%(ctname))
            pos_info[ctname] = ct
        
        rgba=color_palette[ctname]
        ctcolors.append(rgba)
    
    return sank_sid_d, ctlabels, ctcolors

# layout of cell trajectories
def layout_info():
    layout = go.Layout(
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(
                        title="Trajectories",
                        showline=False,
                        showgrid=False,
                        zeroline=False,
                        visible=False,
                    ),
                    xaxis=dict(
                        linewidth=1,
                        linecolor="black",
                    ),
                    yaxis2=dict(
                        # title='Cell counts',
                        domain=[0.0, 0.3],
                        showline=True,
                        showgrid=True,
                        zeroline=True,
                        visible=True,
                        linewidth=1,
                        linecolor="black",
                    ),
                    font_family="Arial",
                    font_size=12,
                    height=500,
                    width=1200,
                    annotations=[
                        go.layout.Annotation(
                            font_size=14,
                            textangle=-90,
                            text="Cell state trajectories",
                            align="left",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=-0.065,
                            y=0.9,
                        ),
                        go.layout.Annotation(
                            font_size=14,
                            textangle=-90,
                            text="Cell counts",
                            align="left",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=-0.065,
                            y=0.03,
                        ),
                    ],
                )
    return layout

# Functions for plotting cell count
def get_trj_plot(sank_sid_d, ctlabels, ctcolors, plot_list):
    trj_fig = go.Sankey(
                        node = dict(
                        pad = 10,
                        thickness = 10,
                        line = dict(color = "black", width = 0.5),
                        label = ctlabels,
                        color = ctcolors,
                        ),
                        link = dict(
                            source = sank_sid_d['source'],
                            target = sank_sid_d['target'],
                            value =  sank_sid_d['gcnt']
                        ),
                        domain = go.sankey.Domain(
                            x = [0.1, 0.9],
                            y = [0.4, 1]
                        ),
                    )
    plot_list.append(trj_fig)
    return plot_list

# functions for plotting cell count
def get_cell_cnt_plot(obj, col_names, sank_sid_d, ctlabels, color_palette, plot_list):
    df = create_cell_cnt(obj)
    df.columns = col_names
    #
    unique_cell_cnt = []
    for i in np.unique(sank_sid_d["source"] + sank_sid_d["target"]):
        x = ctlabels[i]
        if "(" in x:
            x = x[: x.find("(") - 1]
        unique_cell_cnt.append(x)
    unique_cell_cnt = np.unique(unique_cell_cnt)
    
    sel_tct = []
    for ct in unique_cell_cnt:
        if ct in df.index.tolist():
            sel_tct.append(ct)

    l1ct_df = df.loc[sel_tct]
    l1ct_df = pd.melt(l1ct_df.reset_index(), id_vars="index")
    l1ct_df.columns = ["Cell type", "Time", "No. of cells"]
    
    for ct in l1ct_df["Cell type"].unique():
        plot_list.append(
            go.Bar(
                y=l1ct_df[l1ct_df["Cell type"] == ct][
                    "No. of cells"
                ].tolist(),
                x=col_names,
                marker_color=color_palette[ct],
                yaxis="y2",
                name=ct,
                orientation="v",
            )
        )
    return plot_list