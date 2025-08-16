import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import os
import re

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go

import networkx as nx

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import linkage, fcluster

from gutils import calc_gap


# Draw consensus matrix
def draw_ccmatrix(cc, gnames, title='GC', outdir='./'):
    safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)  # 윈도우 호환도 고려
    safe_title = safe_title.replace("/", "_")  # 특히 슬래시 `/` 직접 치환
    linked = linkage(cc, "ward")
    optimal_k = calc_gap(linked)

    clusters = fcluster(linked, optimal_k, criterion="maxclust")
    cluster_df = pd.DataFrame(clusters, index=gnames, columns=["cluster"])

    cmaps = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
         '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

    gid_info = dict(cluster_df['cluster'])

    cluster_colors = [cmaps[gid_info[i]] for i in gnames]

    g = sns.clustermap(cc,
                    cmap="vlag",
                    figsize=(7,7),
                    method="ward",
                    col_colors=cluster_colors,
                    xticklabels=False,
                    yticklabels=False
                    )
    cluster_legend = [mpatches.Patch(color=cmaps[i], label=f'GC {i}')
                  for i in sorted(set(gid_info.values()))]

    g.ax_col_dendrogram.legend(
            handles=cluster_legend,
            title="Clusters",
            loc='upper left',
            bbox_to_anchor=(1, 1.0),
            fontsize=9,
            title_fontsize=10,
            frameon=False
            )

    col_colors_raw = g.col_colors

    color_row = col_colors_raw  # shape: (N,)
    color_row = np.array(color_row)
    col_order = g.dendrogram_col.reordered_ind
    ordered_colors = color_row[col_order]

    boundaries = []
    for color, group in itertools.groupby(enumerate(ordered_colors), key=lambda x: tuple(x[1])):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        boundaries.append((start, end, color))
    
    g.fig.suptitle(safe_title,fontsize=10, y=1.02)

    ax = g.ax_heatmap
    edge_color='white'
    face_alpha=0.07
    for start, end, color in boundaries:
        size = end - start + 1
        rect = mpatches.Rectangle(
            (start, start), size, size,
            linewidth=2.5,
            edgecolor=edge_color,
            facecolor=color + (face_alpha,),
            linestyle='-',
            joinstyle='round'
        )    
        ax.add_patch(rect)
    
    ccoutput = f'{outdir}/ccmatrix/'
    os.makedirs(ccoutput, exist_ok=True)
    g.fig.savefig(f"{ccoutput}/{safe_title}_cm.png", bbox_inches="tight")

def draw_edge_stat(obj):
    if len(obj.rank_dict) == 0:
        print("Perform the edge statistics test first!")
    else:
        dist_df = obj.dist_df.copy()
        
        col_n = max(
            len(np.unique(dist_df["source"].values)),
            len(np.unique(dist_df["target"].values)),
        
        )
        
        color_name = ""
        if col_n > 10:
            color_name = "tab20"
        else:
            color_name = "Set2"

        dist_g = sns.FacetGrid(
            dist_df, row="Interval", col="source", sharey=False, sharex=False,
            hue="target", palette=color_name
        )
        dist_g = dist_g.map(
            sns.kdeplot, "rank_score", fill=False, warn_singular=False
        )
        
        for ax in dist_g.axes.flat:
            row_val = ax.get_title().split('|')[0].split('=')[1].strip()
            col_val = ax.get_title().split('|')[1].split('=')[1].strip()
            ax.set_title(f"{col_val} at Interval {row_val}", fontsize=12)

        os.makedirs(obj.params.output_dir,exist_ok=True)

        dist_g.add_legend(loc='upper left', bbox_to_anchor=(.95,1), title='cell types')
        dist_g.savefig(
            f"{obj.params.output_dir}/{obj.params.output_name}_static_res.pdf"
        )

def draw_transition_graph(obj):
    if len(obj.params.time_point_label) > 1:
        time_points = obj.params.time_point_label
    else:
        time_points = [f'T{str(i)}' for i in range(1, obj.tp_data_num+1)]
        
    interval_conv = dict()
    for i in range(len(time_points)-1):
        time_1 = time_points[i]
        time_2 = time_points[i+1]
        interval_conv[i] = f'{time_1}->{time_2}'                  
            
    # 전체 고유 cell type 추출 (그래프에 등장하는 노드 이름 기준)
    df = obj.pval_df.copy()
    fname = obj.params.answer_path_dir
    if os.path.isfile(fname):
        obj.answer_path = pd.read_csv(fname, sep=",")
        answer_edges = set(zip(obj.answer_path["source"],obj.answer_path["target"]))
        filtered_df = df[df.apply(
            lambda row: (row["source"], row["target"]) in answer_edges, axis=1)]
        df = filtered_df.copy()
        # obj.pval_df = df.copy()
    
    threshold = 0.05
    unique_intervals = df["Interval"].unique()
    interval_graphs = []

    # 각 interval 별로 transition graph 생성
    for interval in sorted(unique_intervals):
        sub_df = df[(df["Interval"] == interval) & (df["p-value"] < threshold)]

        G = nx.DiGraph()
        for _, row in sub_df.iterrows():
            G.add_edge(row["source"], row["target"], weight=-np.log10(row["p-value"]))
        
        interval_graphs.append((interval, G))
        
    all_node_names = set()
    all_weights = []

    for _, G in interval_graphs:
        all_node_names.update(G.nodes())
        all_weights.extend([G[u][v]['weight'] for u, v in G.edges()])

    all_weights = np.array(all_weights)


    # 각 노드에서 cell type 이름 추출
    all_celltypes = list(all_node_names)

    # 자동 색상 생성 (탐색적 colormap 사용)
    cmap = cm.get_cmap("Set2", len(all_celltypes))
    auto_celltype_colors = {
        ct: cmap(i) for i, ct in enumerate(all_celltypes)
    }

    # 시각화 업데이트
    fig, axes = plt.subplots(1, len(interval_graphs), figsize=(6 * len(interval_graphs) + 2, 6))
    if len(interval_graphs) == 1:
        axes = [axes]

    norm = plt.Normalize(vmin=all_weights.min(), vmax=all_weights.max())
    cmap_edges = plt.cm.Reds

    for ax, (interval, G) in zip(axes, interval_graphs):
        pos = nx.shell_layout(G)
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        edge_colors = cmap_edges(norm(weights))

        node_colors = [auto_celltype_colors.get(n) for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                            edgecolors="black", node_size=1200, ax=ax, linewidths=2)

        # 각 node 위치에서 cell type label을 추가하고, 자동으로 충돌 방지
        text_objects = []
        for node, (x, y) in pos.items():
            celltype = node
            # node 바깥쪽에 초기 위치 (오른쪽 아래)
            txt = ax.text(x - 0.2, y - 0.2, celltype, fontsize=9, color='dimgray', style='italic')
            text_objects.append(txt)

        # 자동 조정: 충돌 방지 + 선 추가
        # adjust_text(text_objects, ax=ax, arrowprops=dict(arrowstyle="-", color='lightgray', lw=0.5))
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            edge_cmap=cmap_edges,
            edge_vmin=all_weights.min(),
            edge_vmax=all_weights.max(),
            arrowstyle="-|>",
            arrowsize=15,
            width=2.0,
            alpha=0.9,
            ax=ax,
            connectionstyle="arc3,rad=0.3",
            min_target_margin=15
        )

        ax.set_title(f"{interval_conv[interval]}", fontsize=15, weight="bold")
        ax.set_facecolor("#f7f7f7")
        ax.axis("off")

    # colorbar
    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("-log10(p-value)", fontsize=12)
    plt.tight_layout()
    output_name = f"{obj.params.output_dir}/{obj.params.output_name}_transition_graph.pdf"
    plt.savefig(output_name)
    plt.show()

def draw_patterns(obj):
    from gutils import l2norm, vect_mu, convert_path_name, make_gene_set_frame
    from gutils import plotting_patterns

    print("Plotting time-series gene expression patterns...")

    output_name = f"{obj.params.output_dir}/{obj.params.output_name}_patterns.pdf"
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_name)

    # Figure positions
    row_n, col_n, pos = 3, 3, 0
    fig, ax = plt.subplots(figsize=(14,10), nrows=row_n, ncols=col_n)

    # Trajectory keys
    pt_keys = list(obj.merge_pattern_dict.keys())
    time_len = obj.tp_data_num

    # Gene set data frame
    gene_set_df = pd.DataFrame()

    # Plotting time-series gene expression patterns
    fc_th = 1.5
    for idx, key in enumerate(pt_keys):
        # Normalization
        pt_df = l2norm(obj.merge_pattern_dict[key])
        cent = vect_mu(pt_df)

        if len(pt_df.columns) != time_len: continue
        if max(cent) / min(cent) < fc_th: continue

        # Customizing a personalized list of specific time points for each user
        if len(obj.params.time_point_label) != 0:
            pt_df.columns = obj.params.time_point_label
        
        # Store cell-state trajectory info and gene sets
        convert_name = convert_path_name(obj, key)
        start_cells = convert_name[: convert_name.find("-")]
        gene_set_df = make_gene_set_frame(idx, gene_set_df, pt_df, key, convert_name)

        # Update position
        if pos // col_n == col_n:
            fig.tight_layout()
            pdf.savefig(fig)
            plt.show()
            fig.clf()
            fig, ax = plt.subplots(figsize=(14,10), nrows=row_n, ncols=col_n)
            pos = 0
        
        # Plotting gene expression patterns
        plotting_patterns(pt_df, key, start_cells, ax, pos)
        pos+=1
    
    # Store gene set information for cell trajectory
    gene_set_df.to_csv(f'{obj.params.output_dir}/{obj.params.output_name}_pattern_genes.csv',sep=",")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.show()
    pdf.close()

def draw_trajectories(obj):
    print("Plotting cell-state trajectories...")
    from gutils import get_celltype_edges, get_gtra_res, generate_palette_dict
    from gutils import get_plot_dat, get_trj_plot, get_cell_cnt_plot, layout_info
    celltype_info, celltypes_sb, celltypes_bs = get_celltype_edges(obj)

    sdat, trtp_celltypes, sub_celltypes = get_gtra_res(obj, celltypes_sb)
    start_cells = [i.split("_")[1] for i in trtp_celltypes if int(i[i.find("t") + 1 : i.find("_")]) == 0]

    # indexing tr celltypes
    trtp_celltypes_d = {x: -1 for x in trtp_celltypes}
    ctidx = 0
    for k in trtp_celltypes_d.keys():
        trtp_celltypes_d[k] = ctidx
        ctidx += 1

    # loading color palette for cell types
    color_palette = generate_palette_dict(celltype_info)
    sid = obj.params.output_name

    for target_ct in start_cells:
        # Flag
        inter = len(set(celltypes_bs[target_ct]).intersection(sdat.celltypes))
        if not inter:
            continue

        # meta info
        tp = sdat.tp
        col_names = ["T%s" % (x) for x in range(1, tp + 1)]
        sank_sid_d = dict()
        csid = None

        sank_sid_d, ctlabels, ctcolors = get_plot_dat(
            sdat,
            trtp_celltypes,
            trtp_celltypes_d,
            celltypes_bs,
            target_ct,
            color_palette,
        )

        plot_list = []

        print("plotting sankey plots of...")

        #### DRAWING CELL TRAJECTORY ####
        plot_list = get_trj_plot(sank_sid_d, ctlabels, ctcolors, plot_list)

        #### DRAWING CELL COUNTS ####
        plot_list = get_cell_cnt_plot(
            obj, col_names, sank_sid_d, ctlabels, color_palette, plot_list
        )

        layout = layout_info()

        fig = go.Figure(
            data=plot_list,
            layout=layout,
        )

        fig.update_layout(
            title_text="[ %s cell ] trajectories" % (target_ct),
            legend=dict(yanchor="top", y=1.1, xanchor="right", x=1.1),
        )
        fig_out = f"{obj.params.output_dir}/sankey_plot"
        os.makedirs(fig_out, exist_ok=True)
        if "/" in target_ct:
            target_ct = target_ct.replace("/", "_")
        fig.write_image("%s/%s_%s.pdf" % (fig_out, sid, target_ct))
        fig.show()
    


