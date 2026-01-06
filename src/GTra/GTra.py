import numpy as np
import pandas as pd
import scanpy as sc

import anndata as ad
import networkx as nx

from tqdm import tqdm

from scipy.stats import wilcoxon
from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix

from .preproc import filter_genes, concat_meta
from .gutils import JS, JS_threshold_test, cal_cos_dist, get_edge_info, cal_rank, get_networks, check_standard_path


class GTraObject(object):
    class GTraParam:
        __slots__ = (
            "low_gene_num", "low_cell_num", "mito_percent", "filter_cell_n", "hvg_n",
            "norm_flag", "label_flag", "gene_norm_flag", "cn_neighbors", "gn_neighbors", 
            "cn_cluster_resolution", "gn_cluster_resolution", "dist_threshold", "sw", "dw",
            "kw","top_rank", "static_th", "answer_path_type", "answer_path_dir",
            "cell_type_label", "time_point_label", "output_dir", "output_name"
        )

        # Parameters of GTraParam
        def __init__(self):
            # Data preprocessing parameters
            self.low_gene_num = 200
            self.low_cell_num = 3
            self.mito_percent = 0.2
            self.hvg_n = 2000
            self.norm_flag = True

            # STEP 1: GTra's clustering
            self.label_flag = False # Cell type label info
            self.cn_neighbors = 10 # KNN (cells)
            self.gn_neighbors = 15 # KNN (genes)
            self.cn_cluster_resolution = 3 # Leiden resolution
            self.gn_cluster_resolution = 0.3 # Leiden resolution
            self.gene_norm_flag = True
            self.filter_cell_n = 10

            # STEP 2: Select candidate edges and construct trajectory
            self.dist_threshold = 0.5 # consine distance
            self.sw = 1
            self.dw = 3
            self.kw = 5
            self.top_rank = 10 # # of candidiate edges
            self.answer_path_type = "" # Answer path type (e.g. PBMC, NEURON)
            self.answer_path_dir = "" # Answer path directory
            self.cell_type_label = ""
            self.static_th = 90 # Threshold of statical testing

            # Customizing parameters
            self.time_point_label = [] # (e.g. day+1, day+3, ...)
            self.output_dir = "./"
            self.output_name = "GTra"
        
    # Parameters of GTraObject
    def __init__(self):
        self.params = self.GTraParam()

        # Parameters for STEP 1
        self.tp_data_dict = dict()
        self.tp_data_num = 0
        self.cell_cluster_label = dict()
        self.cell_optimal_k = dict()
        self.cell_label_index = dict()
        self.genes = dict()
        self.tp_genes_dict = dict()
        self.gene_label_info = dict()
        self.ccmatrix = dict()
        self.celltype_colors = dict()

        # Parameters for STEP 2
        self.ctc_list = dict()
        self.gtg_list = dict()
        self.edge_info = dict()
        self.selected_edges = dict()
        self.f = lambda x: x
        self.node_info = pd.DataFrame(columns=["from", "to", "sim", "cos_dist", "rank_val"])

        self.node_cnt = 0
        self.net_info = [[[]]]
        self.answer_path = pd.DataFrame() 
        self.cnt_dict = dict()
        self.rank_dict = dict()
        self.candidate_dict = dict()
        self.dist_df = pd.DataFrame()
        self.pval_df = pd.DataFrame()
        self.pval_th = 5e-2
        self.static_flag = False

        # Parameters for STEP 3
        self.path_genes = dict()
        self.path_gene_sets = dict()
        self.path_candidate = dict()
        self.merge_pattern_dict = dict()
        self.merge_pattern_within_distance = dict()
        self.merge_path_dict = dict()
        self.cluster_centers = dict()

        # Parameters for STEP 4
        self.cell_type_info = pd.DataFrame()
        self.merge_node_info = pd.DataFrame.from_records(
            list(map(self.f, [])), columns = ["from", "to"]
        )
        self.merge_net_info = [[[]]]
    
    # Upload time-series scRNA-seq data
    def upload_time_scRNA(self, *args):
        if len(args) == 2: # args[0]: dataframe, args[1]: obs
            self.params.label_flag = True
            adata = sc.AnnData(args[0], obs=args[1])
        else:
            adata = sc.AnnData(args[0])
        
        if self.tp_data_num == 0:
            self.genes = adata.var_names.tolist()

        self.tp_data_dict[self.tp_data_num] = adata
        self.tp_data_num += 1
        

    # Filtering low expressed genes
    def select_genes(self):
        fgenes = filter_genes(ad.concat(self.tp_data_dict))
        for time in range(self.tp_data_num):
            X = self.tp_data_dict[time][:, fgenes].copy()

            if time == 0: self.genes = fgenes

            self.tp_data_dict[time] = X

    ##########################################################
    ## ------- STEP 1: GTra's clustering ------------------ ##
    ##########################################################

    # Perform consensus clustering (GTra)
    def cc_clustering(self, N=30):
        from .cluster_func import parallel_testing, get_cc_clusters
        parallel_testing(self, N=N)
        get_cc_clusters(self)

    #############################################################
    ## ------- STEP 2: Construct cell-state trajectory ------- ##
    #############################################################

    # Calculate edge score
    def cal_edge_score(self, tp1, tp2):
        # Load scRNA data
        t1_df = self.tp_data_dict[tp1].to_df().T
        t2_df = self.tp_data_dict[tp2].to_df().T

        # Load gene sets
        t1_genes = self.gene_label_info[tp1]
        t2_genes = self.gene_label_info[tp2]

        # Sample and gene cluster index (init)
        t1_s, t1_g, t2_s, t2_g = 0, 0, 0, 0

        # Get optimal threshold
        optimal_th = JS_threshold_test(self, tp1, tp2)

        # Comparison intersected gene set between adjacent time points
        edge_info = []
        for t1_s in range(len(t1_genes)):
            cos_dists = []
            dists = {}
            einfo = {}
            for t1_g in range(len(t1_genes[t1_s])):
                for t2_s in range(len(t2_genes)):
                    for t2_g in range(len(t2_genes[t2_s])):
                        sim, inter_genes = JS(
                            t1_genes[t1_s][t1_g], t2_genes[t2_s][t2_g]
                        )
                        if sim < optimal_th:
                            continue

                        # dist, kl_sim = cal_cos_dist(
                        #     self, tp1, tp2, t1_s, t2_s, t1_df, t2_df, inter_genes
                        # )
                        dist = cal_cos_dist(
                            self, tp1, tp2, t1_s, t2_s, t1_df, t2_df, inter_genes
                        )
                        if dist == -1:
                            continue
                        # if kl_sim < 0.05:
                        #     continue

                        # Within distance of edge
                        cent = np.mean(dist)

                        if dists.get(t2_s) is None:
                            dists[t2_s] = []
                            einfo[t2_s] = []

                        dists[t2_s].append(dist)
                        # einfo[t2_s].append([t1_s, t1_g, t2_s, t2_g, sim, cent, kl_sim])
                        einfo[t2_s].append([t1_s, t1_g, t2_s, t2_g, sim, cent])
                        cos_dists.append(dist)

            # Population
            pop_cos = np.array(sum(cos_dists, []))

            # Statistical test (non-parametric test) + FDR correciton
            ## 2025-10-29 code modification
            num_edges = 0
            all_pvals = []
            edge_refs = []
            for k in dists:
                for i, dat in enumerate(dists[k]):
                    if len(dat) < 2 or np.allclose(dat, np.mean(pop_cos)):
                        continue
                    _, pval = wilcoxon(np.array(dat)-np.mean(pop_cos), zero_method='zsplit')
                    num_edges +=1
                    all_pvals.append(pval)
                    edge_refs.append((k, i))

            from statsmodels.stats.multitest import multipletests
            # Skip if no valid p-values
            if len(all_pvals) == 0: continue

            rejected, adj_pvals, _, _ = multipletests(all_pvals, alpha=0.05, method='fdr_bh')
            for (k, i), adj_p, reject in zip(edge_refs, adj_pvals, rejected):
                if not reject:
                    continue
                z = np.mean(pop_cos) - np.mean(dists[k][i])
                if z < 0:
                    continue
                einfo[k][i].extend([adj_p])
                edge_info.append(einfo[k][i])


            # num_edges = 0
            # p_dict = {}
            # for k in dists:
            #     p_dict[k] = {}
            #     for i, dat in enumerate(dists[k]):
            #         _, pval = wilcoxon(
            #             np.array(dat) - np.mean(pop_cos), zero_method="zsplit"
            #         )
            #         num_edges += 1
            #         p_dict[k][i] = pval

            # # Calculate adjusted p-value
            # for k in p_dict:
            #     for i, p in p_dict[k].items():
            #         z = np.mean(pop_cos) - np.mean(dists[k][i])
            #         # Bonfferoni correction (adj P-val)
            #         adj_p = min(p * num_edges, 1.0)

            #         if (z < 0) | (adj_p > 0.05):
            #             continue

            #         einfo[k][i].extend([adj_p])
            #         edge_info.append(einfo[k][i])

        self.edge_info[tp1] = edge_info

    # Rank test for candidate edges
    def edge_rank_test(self, tp1, tp2):
        # Get edge info for previous time point data
        edge_info_dict = get_edge_info(self, tp1)

        # Edge candidates that have passed statistical tests
        if len(self.candidate_dict):
            candidated_edges = self.candidate_dict[tp1]
            check_edges = {}
            for i in candidated_edges:
                tok = i.split("_")
                s, t = int(tok[0]), int(tok[1])
                if check_edges.get(s) is None:
                    check_edges[s] = [t]
                else:
                    check_edges[s].append(t)

        # Ranking candidate edges [Modi: 25-11-21]
        RANK_PARAM = -1
        edge_info_results = []
        for t1_s, edge_info in edge_info_dict.items():
            conv_edge = cal_rank(edge_info, self.params.sw, self.params.dw, self.params.kw)
            sort_conv_edge = np.array(sorted(conv_edge, key=lambda x: x[RANK_PARAM]))

            # Edge candidates that have passed statistical tests
            if len(self.candidate_dict) and (check_edges.get(t1_s)):
                sort_conv_edge = np.array(
                    [list(i) for i in sort_conv_edge if i[1] in check_edges[t1_s]]
                )

            top_rank = min(self.params.top_rank, len(sort_conv_edge))

            # If answer path information exist then ~
            if (self.static_flag == True):
                edge_info_results = check_standard_path(
                    self, tp1, t1_s, top_rank, sort_conv_edge, edge_info_results
                )
            else:
                for rank in range(top_rank):
                    label_info = list(map(int, sort_conv_edge[rank, :3]))
                    score_info = list(sort_conv_edge[rank, 3:])
                    edge_info_results.append([t1_s] + label_info + score_info)

        # Convert edge info name
        conv_edge_info = []
        for einfo in edge_info_results:
            source = f"t{str(tp1)}_{str(einfo[0])}_{str(einfo[1])}"
            target = f"t{str(tp2)}_{str(einfo[2])}_{str(einfo[3])}"
            conv_edge_info.append([source, target, einfo[4], einfo[5], einfo[-1]])

        # Save candidate edges
        self.selected_edges[tp1] = conv_edge_info
    
    # Select candidate edges
    def select_candidate_edges(self):
        self.cell_type_info = concat_meta(self)
        display_name = "Construct cell-state trajectories..."
        for tp in tqdm(
            range(self.tp_data_num-1),
            total = self.tp_data_num-1,
            desc=display_name,
            ncols=100,
            ascii=" =",
            leave=True
            ):
            self.cal_edge_score(tp, tp+1)
            self.edge_rank_test(tp, tp+1)
        
        # Save edge info
        records = []
        for tp in range(self.tp_data_num-1):
            records.extend(self.selected_edges[tp])
        self.node_info = pd.DataFrame(records, columns=["from","to","sim","cos_dist","rank_val"])
        
    # Construct cell-state trajectories
    def construct_trajectory(self):
        self.select_candidate_edges()

        # Create sub-graphs
        g = nx.from_pandas_edgelist(
            self.node_info, "from", "to", create_using=nx.DiGraph()
            )
        sub_graphs = list(g.subgraph(c) for c in nx.weakly_connected_components(g))

        # Get candidate path info
        self.net_info = get_networks(self, sub_graphs)

    ################################################################
    ## ------- STEP 3: Gene expression pattern clustering ------- ##
    ################################################################

    # Time-series pattern clustering
    def pattern_clustering(self):
        from .gutils import group_cell_type_trajectory, merge_expr, elbow_method
        from .gutils import pattern_filtering, merge_sim_patterns, renaming_pattern_id, save_pattern_centroid

        self.merge_path_dict = group_cell_type_trajectory(self.net_info)

        path_label = 0
        displays = "Step 3: Pattern clustering..."
        mpdv = list(self.merge_path_dict.values())
        for i in tqdm(
            range(len(mpdv)),
            total=len(mpdv),
            desc=displays,
            ncols=100,
            ascii=" =",
            leave=True,
        ):
            cluster_label = 0
            paths = mpdv[i]
            for path in paths:
                key = f"{str(path_label)}_{str(cluster_label)}"
                expr = merge_expr(self, path)

                if len(expr) < 20:
                    self.merge_pattern_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue

                try:
                    # Select optimal K using elbow method
                    optimal_k = elbow_method(expr)
                    # Perform clustering
                    spk = SphericalKMeans(
                        n_clusters=optimal_k, random_state=25, max_iter=100
                    ).fit(csr_matrix(expr))
                except:
                    self.merge_pattern_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue

                # Save clustered pattern info
                for k in range(optimal_k):
                    gene_cluster = [i for i, c in enumerate(spk.labels_) if c == k]
                    key_k = f"{key}_{str(k)}"
                    self.merge_pattern_dict[key_k] = expr.iloc[gene_cluster, :]
                    self.cluster_centers[key_k] = spk.cluster_centers_

                cluster_label += 1
            path_label += 1

        # Filtering low qulaity patterns
        self.merge_pattern_dict = pattern_filtering(self)
        # Merge selected patterns
        new_mp_dict = merge_sim_patterns(self)
        # Convert pattern name
        self.merge_pattern_dict = renaming_pattern_id(new_mp_dict)
        # Save pattern centroid
        save_pattern_centroid(self)

    # ----------------- Visualization functions  -----------------##

    # Plotting consensus matrix
    def plot_cluster_matrix(self):
        from .visualize import draw_ccmatrix
        out_dir = self.params.output_dir
        for tp, cl in self.ccmatrix.items():
            df = self.tp_data_dict[tp].obs.drop_duplicates()
            tp_ctdict = dict(zip(df["cluster_label"], df[self.params.cell_type_label]))
            for key in cl.keys():
                ct = tp_ctdict[key]
                title = f'T{tp}_{ct}'
                cc = self.ccmatrix[tp][key]
                draw_ccmatrix(cc, gnames=self.genes, title=title, outdir=out_dir)
    
    # Plotting edge static results
    def plot_edge_stat(self):
        from .visualize import draw_edge_stat
        draw_edge_stat(self)
    
    # Plotting cell-state transition graph
    def plot_transition_graph(self):
        from .visualize import draw_transition_graph
        draw_transition_graph(self)

    
    # Plotting gene-gene transition matrix
    def plot_gg_matrix(self):
        from .visualize import draw_gg_matrix
        draw_gg_matrix(self)

    # Plotting expression patterns
    def plot_expressions(self):
        from .visualize import draw_patterns
        draw_patterns(self)
    
    # Plotting cell-state trajectories
    def plot_trajectory(self):
        from .visualize import draw_trajectories
        draw_trajectories(self)
