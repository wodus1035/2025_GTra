import numpy as np
import pandas as pd
import scanpy as sc

import anndata as ad
import networkx as nx

from tqdm import tqdm

from soyclustering import SphericalKMeans
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity

from preproc import *
from gutils import *
from cluster_func import *

class GTraObject(object):
    class GTraParam:
        # __slots__ = (
        #     "low_gene_num", "low_cell_num", "mito_percent", "hvg_n", "norm_flag",
        #     "label_flag", "gene_norm_flag", "cn_neighbors", "gn_neighbors", "cn_cluster_resolution",
        #     "gn_cluster_resolution", "filter_cell_n", "dist_threshold", "sw", "dw", "kw", "top_rank",
        #     "static_th", "answer_path_type", "answer_path_dir", "cell_type_label", "time_point_label",
        #     "output_dir","output_name")

        # Basic parameters
        def __init__(self):
            # For data preprocessing
            self.low_gene_num = 200
            self.low_cell_num = 3
            self.mito_percent = 0.2
            self.hvg_n = 2000
            self.norm_flag = True
            
            # Step 1: GTra's consensus clustering
            self.label_flag = False
            self.gene_norm_flag = True
            self.cn_neighbors = 10 # For cells (kNN)
            self.gn_neighbors = 15 # For genes (kNN)
            self.cn_cluster_resolution = 0.5
            self.gn_cluster_resolution = .3
            self.filter_cell_n = 10

            # Step 2: Select candidate edges and construct trajectories
            self.dist_threshold = .5 # For cosine distance
            self.sw, self.dw, self.kw = .5, .5, .5 # sw: jaccard, dw: cosine, kw: KL
            self.top_rank = 10 # candidate's edges
            self.static_th = 90 # threshold of statistical testing
            self.answer_path_type = "" # organism or dataset
            self.answer_path_dir = "" # directory contained answer path info
            self.cell_type_label = ""
            self.time_point_label = [] # [day+1, day+5, ...]
            self.min_pattern_gene = 20

            # Save directory
            self.output_dir = "./"
            self.output_name = "GTra"
    
    # GTraObject parameters
    def __init__(self):
        self.params = self.GTraParam()

        ## =========== Step 1's parameters =========== ##
        self.tp_data_dict = {}
        self.tp_data_num = 0 # total count of time points
        # For cell clustering
        self.cell_cluster_label = {}
        self.cell_optimal_k = {}
        self.cell_label_index = {}
        self.celltype_colors = {}

        # For gene clustering
        self.genes = {}
        self.tp_genes_dict = {}
        self.gene_label_info = {}
        self.ccmatrix = {}

        ## =========== Step 2's parameters =========== ##
        self.ctc_list, self.gtg_list = {}, {}
        self.edge_info = {}
        self.selected_edges = {}
        self.node_info, self.node_cnt = pd.DataFrame(), 0
        self.net_info = [[[]]]
        self.answer_path = pd.DataFrame() # standard trajectory info [source -> target]
        self.cnt_dict, self.score_dict = {}, {} # statistical testing
        self.candidate_dict = {} # edges passed statistical testing
        self.dist_df, self.pval_df = pd.DataFrame(), pd.DataFrame()
        self.pval_th = 0.05
        self.static_flag = False

        ## =========== Step 3's parameters =========== ##
        self.path_gene_sets = {}
        self.path_candidates = {}
        self.merge_pattern_dict = {}
        self.merge_pattern_within_distance= {}
        self.merge_path_dict = {}
        self.cluster_centers = {}
        self.cell_type_info = pd.DataFrame()
        self.f = lambda x: x
        self.merge_node_info = pd.DataFrame.from_records(
            list(map(self.f, [])), columns = ["from", "to"]
        )
        self.merge_net_info = [[[]]]
    
    # Upload time-series scRNA-seq dataset
    def upload_time_scRNA(self, *args):
        """
        Upload scRNA-seq data for each time point.
        Ensures all time points use the same gene list and same ordering.
        """
        if len(args) == 2: # args[0]: matrix, args[1]: obs
            self.params.label_flag = True # celltype label check
            adata = sc.AnnData(args[0], obs=args[1])
        else:
            adata = sc.AnnData(args[0])
        
        if self.tp_data_num == 0:
            self.genes = adata.var_names.tolist()
        
        self.tp_data_dict[self.tp_data_num] = adata
        self.tp_data_num += 1

    # Filtering low-expressed genes
    def select_genes(self):
        """
        Select filtered genes and enforce a unified gene set across all time points.

        Ensures that every time point uses the same genes in the same order.
        Missing genes in any time point are automatically added as zero vectors.
        """

        gene_sets = [set(filter_genes(self.tp_data_dict[tp]))
                     for tp in range(self.tp_data_num)]
        fgenes = list(gene_sets[0].intersection(*gene_sets[1:]))
        
        for tp in range(self.tp_data_num):
            adata = self.tp_data_dict[tp]
            missing = set(fgenes) - set(adata.var_names)
            if missing:
                zeros = np.zeros((adata.n_obs, len(missing)))
                X = np.hstack([adata.X.toarray(), csr_matrix(zeros)])
                new_genes = list(adata.var_names) + list(missing)
                X = X[:, [new_genes.index(g) for g in fgenes]]
                self.tp_data_dict[tp] = sc.AnnData(X, obs=adata.obs, 
                                                   var=pd.DataFrame(index=fgenes))
            else:
                self.tp_data_dict[tp] = adata[:, fgenes]
        self.genes = fgenes

    ###############################################################################
    ## ============ Step 1: Identifying cell type-specific clusters ============ ##
    ###############################################################################

    # Perform consensus clustering
    def find_gclusters(self, N=50):
        statistical_testing(self, N=N)
        cc_clustering(self)
        create_color_dict(self)
    
    ###########################################################################
    ## ============= Step 2: Construct cell-state trajectories ============= ##
    ###########################################################################
    
    # Calculate transition scores between adjacent time points
    def cal_edge_score(self, tp1, tp2):
        """
        Optimized, vectorized computation of edge scores between time points.
        """
        
        # ------------------------------------------------------
        # Load Expression Data (cell x gene)
        # ------------------------------------------------------
        A1 = self.tp_data_dict[tp1].to_df()
        A2 = self.tp_data_dict[tp2].to_df()

        A1_z = vector_norm(A1)
        A2_z = vector_norm(A2)
        
        # gene clusters at each timepoint
        GM1 = flatten_gene_modules(self.gene_label_info[tp1])
        GM2 = flatten_gene_modules(self.gene_label_info[tp2])
        
        # JS matrix
        js_matrix = compute_js_matrix(GM1, GM2, self.genes)
        optimal_th = JS_threshold_test(self, tp1, tp2)

        
        # Cluster mean expressions
        mean1 = compute_cluster_means(A1_z, self.cell_label_index[tp1])
        mean2 = compute_cluster_means(A2_z, self.cell_label_index[tp2])
        gene_index = {g:i for i, g in enumerate(A1.columns)}

        edges = []

        for i, (c1, m1, g1_list) in enumerate(GM1):
            for j, (c2, m2, g2_list) in enumerate(GM2):

                js = js_matrix[i, j]
                if js < optimal_th:
                    continue

                # shared gene 기반 vector 추출
                v1, v2, shared_n = extract_shared_vectors(
                    g1_list, g2_list,
                    mean1[c1], mean2[c2],
                    gene_index
                )
                if shared_n == 0:
                    continue

                # cosine
                cos = centered_cosine(v1, v2)
                if cos is None:
                    continue

                # JS divergence (stable KL)
                jsdiv = js_divergence(v1, v2)

                # joint score
                score = compute_joint_score(js, cos, jsdiv)
                if score is None:
                    continue

                edges.append([
                    c1, m1, c2, m2,
                    float(js), float(cos), float(jsdiv), float(score)
                ])
        
        self.edge_info[tp1] = edges
    
    # Select candidate edges with statistical testing and answer path info
    def edge_score_test(self, tp1, tp2):
        """
        edges: list of [c1, m1, c2, m2, js, cos, jsdiv, score]
        """
        edges = self.edge_info[tp1]
        if len(edges) == 0:
            self.selected_edges[tp1] = []
            return
        
        # Edge candidates that have passsed statistical testing
        if len(self.candidate_dict):
            candidated_edges = self.candidate_dict[tp1]
            check_edges = {}
            for i in candidated_edges:
                tok = i.split('_')
                s, t = int(tok[0]), int(tok[1])
                if check_edges.get(s) is None:
                    check_edges[s] = [t]
                else:
                    check_edges[s].append(t)
        
        # ---- 1) Score based alignment ---- #
        edges_sorted = sorted(edges, key=lambda x: x[-1], reverse=True)
        
        # ---- 2) Top-k selection ---- #
        edges_top = select_by_percentile(edges_sorted,pct=0.25)

        # ---- 3) Answer path filtering (선택사항) ---- #
        if self.static_flag:
            edges_top = filter_by_answer_path(self, tp1, edges_top)

        # ---- 4) Edge name match ---- #
        final_edges = []
        for e in edges_top:
            c1, m1, c2, m2 = e[:4]
            source = f"t{tp1}_{c1}_{m1}"
            target = f"t{tp2}_{c2}_{m2}"
            final_edges.append([source, target]+e[4:])

        self.selected_edges[tp1] = final_edges
    
    # Select edges
    def select_candidate_edges(self):
        self.cell_type_info = concat_meta(self)
        display_name = "Construct cell-state trajectories.."
        for tp in tqdm(
            range(self.tp_data_num-1),
            total=self.tp_data_num-1,
            desc=display_name,
            ncols=100,
            ascii=" =",
            leave=True
        ):
            self.cal_edge_score(tp, tp+1)
            self.edge_score_test(tp, tp+1)
        
        # Save edge info
        records = []
        for tp in range(self.tp_data_num - 1):
            records.extend(self.selected_edges[tp])
        
        self.node_info = pd.DataFrame(records, columns=[
            "from", "to", "sim", "cos", "js_div", "Escore"
        ])
    
    # Construct cell-state trajectories
    def construct_trajectories(self):
        self.select_candidate_edges()
        
        # Create sub-graphs
        g = nx.from_pandas_edgelist(
            self.node_info, "from", "to", create_using=nx.DiGraph()
        )
        
        sub_graphs = list(g.subgraph(c) for c in nx.weakly_connected_components(g))
        
        # Get candidate path info
        self.net_info = get_networks(self, sub_graphs)
            
    
    ############################################################################
    ## ============= Step 3: Gene expression pattern clustering ============= ##
    ############################################################################

    # from gutils import group_cell_type_trajectory, merge_expr, elbow_method

    # Time-series pattern clustering
    def pattern_clustering(self):

        self.merge_path_dict = group_cell_type_trajectory(self.net_info)

        path_label = 0
        displays = "Time-series pattern clustering..."
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

                if len(expr) < self.params.min_pattern_gene:
                    self.merge_pattern_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue
                try:
                    optimal_k = elbow_method(expr)
                    spk = SphericalKMeans(
                        n_clusters=optimal_k, random_state=25, max_iter=100
                    ).fit(csr_matrix(expr))
                
                except:
                    self.merge_path_dict[key] = expr
                    self.cluster_centers[key] = expr.mean().values
                    continue

                for k in range(optimal_k):
                    gene_cluster = [i for i, c in enumerate(spk.labels_) if c == k]
                    key_k = f"{key}_{str(k)}"
                    self.merge_pattern_dict[key_k] = expr.iloc[gene_cluster, :]
                    self.cluster_centers[key_k] = spk.cluster_centers_
            
                cluster_label += 1
            path_label += 1
    
        self.merge_pattern_dict = pattern_filtering(self) # Filtering low-quality patterns
        new_mp_dict = merge_sim_patterns(self) # Merge selected patterns
        self.merge_pattern_dict = renaming_pattern_id(new_mp_dict) # Convert pattern name
        save_pattern_centroid(self) # Save pattern centroid    