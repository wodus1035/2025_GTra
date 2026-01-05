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

from preproc import create_color_dict
def draw_edge_stat(obj):
    if len(obj.rank_dict) == 0:
        print("Perform the edge statistics test first!")
    else:
        create_color_dict(obj)
        dist_df = obj.dist_df.copy()
        dist_g = sns.FacetGrid(
            dist_df, row="Interval", col="source", sharey=False, sharex=False,
            hue="target", palette=obj.celltype_colors
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

# def draw_transition_graph(obj):
#     if len(obj.params.time_point_label) > 1:
#         time_points = obj.params.time_point_label
#     else:
#         time_points = [f'T{str(i)}' for i in range(1, obj.tp_data_num+1)]
        
#     interval_conv = dict()
#     for i in range(len(time_points)-1):
#         time_1 = time_points[i]
#         time_2 = time_points[i+1]
#         interval_conv[i] = f'{time_1}->{time_2}'                  
            
#     # 전체 고유 cell type 추출 (그래프에 등장하는 노드 이름 기준)
#     df = obj.pval_df.copy()
#     fname = obj.params.answer_path_dir
#     if os.path.isfile(fname):
#         obj.answer_path = pd.read_csv(fname, sep=",")
#         answer_edges = set(zip(obj.answer_path["source"],obj.answer_path["target"]))
#         filtered_df = df[df.apply(
#             lambda row: (row["source"], row["target"]) in answer_edges, axis=1)]
#         df = filtered_df.copy()
#         # obj.pval_df = df.copy()
    
#     threshold = 0.05
#     unique_intervals = df["Interval"].unique()
#     interval_graphs = []

#     # 각 interval 별로 transition graph 생성
#     for interval in sorted(unique_intervals):
#         sub_df = df[(df["Interval"] == interval) & (df["p-value"] < threshold)]

#         G = nx.DiGraph()
#         for _, row in sub_df.iterrows():
#             G.add_edge(row["source"], row["target"], weight=-np.log10(row["p-value"]))
        
#         interval_graphs.append((interval, G))
        
#     all_node_names = set()
#     all_weights = []

#     for _, G in interval_graphs:
#         all_node_names.update(G.nodes())
#         all_weights.extend([G[u][v]['weight'] for u, v in G.edges()])

#     all_weights = np.array(all_weights)


#     # 각 노드에서 cell type 이름 추출
#     # all_celltypes = list(all_node_names)

#     # 자동 색상 생성 (탐색적 colormap 사용)
#     # cmap = cm.get_cmap("Set2", len(all_celltypes))
#     # auto_celltype_colors = {
#     #     ct: cmap(i) for i, ct in enumerate(all_celltypes)
#     # }
#     auto_celltype_colors = obj.celltype_colors

#     # 시각화 업데이트
#     fig, axes = plt.subplots(1, len(interval_graphs), figsize=(6 * len(interval_graphs) + 2, 6))
#     if len(interval_graphs) == 1:
#         axes = [axes]

#     norm = plt.Normalize(vmin=all_weights.min(), vmax=all_weights.max())
#     cmap_edges = plt.cm.Reds

#     for ax, (interval, G) in zip(axes, interval_graphs):
#         pos = nx.shell_layout(G)
#         weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
#         edge_colors = cmap_edges(norm(weights))

#         node_colors = [auto_celltype_colors.get(n) for n in G.nodes()]

#         nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
#                             edgecolors="black", node_size=1200, ax=ax, linewidths=2)

#         # 각 node 위치에서 cell type label을 추가하고, 자동으로 충돌 방지
#         text_objects = []
#         for node, (x, y) in pos.items():
#             celltype = node
#             # node 바깥쪽에 초기 위치 (오른쪽 아래)
#             txt = ax.text(x - 0.2, y - 0.2, celltype, fontsize=9, color='dimgray', style='italic')
#             text_objects.append(txt)

#         # 자동 조정: 충돌 방지 + 선 추가
#         # adjust_text(text_objects, ax=ax, arrowprops=dict(arrowstyle="-", color='lightgray', lw=0.5))
#         nx.draw_networkx_edges(
#             G, pos,
#             edge_color=edge_colors,
#             edge_cmap=cmap_edges,
#             edge_vmin=all_weights.min(),
#             edge_vmax=all_weights.max(),
#             arrowstyle="-|>",
#             arrowsize=15,
#             width=2.0,
#             alpha=0.9,
#             ax=ax,
#             connectionstyle="arc3,rad=0.3",
#             min_target_margin=15
#         )

#         # The above code is setting the title of a plot in a matplotlib figure. It is using the value
#         # of the `interval` variable as a key to access a corresponding value in the `interval_conv`
#         # dictionary, and then formatting that value as the title of the plot. The title will be
#         # displayed with a font size of 15 and bold weight.
#         ax.set_title(f"{interval_conv[interval]}", fontsize=15, weight="bold")
#         ax.set_facecolor("#f7f7f7")
#         ax.axis("off")

#     # colorbar
#     divider = make_axes_locatable(axes[-1])
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, cax=cax)
#     cbar.set_label("-log10(p-value)", fontsize=12)
#     plt.tight_layout()
#     output_name = f"{obj.params.output_dir}/{obj.params.output_name}_transition_graph.pdf"
#     plt.savefig(output_name)
#     plt.show()

# 2025-11-04: Update plot (draw_transition_graph)
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import numpy as np

def draw_true_self_loop(ax, x, y, color="#333333", radius=0.15,
                        angle=45, lw=2, alpha=0.9, weight=None):
    """
    Draw a true circular self-loop arrow around node (x, y).
    weight 값에 따라 선 두께를 자동 조정합니다.
    """
    Path = mpath.Path
    theta = np.deg2rad(angle)

    # 🎯 weight 기반 두께 조정
    if weight is not None:
        lw = np.clip(0.5 + 0.4 * weight, 1.0, 5.0)  # log10 기반 스케일

    # 루프 중심 계산
    cx = x + radius * np.cos(theta)
    cy = y + radius * np.sin(theta)

    # 반원형 arc
    arc = mpatches.Arc((cx, cy), radius*2, radius*2, angle=0,
                       theta1=180, theta2=540,
                       color=color, lw=lw, alpha=alpha, zorder=6)
    ax.add_patch(arc)

    # 화살표 머리 (반원 끝)
    arrow_angle = theta - np.pi / 2
    arrow_x = cx + radius * np.cos(arrow_angle)
    arrow_y = cy + radius * np.sin(arrow_angle)
    ax.arrow(arrow_x, arrow_y, 0.001, 0.001,
             head_width=0.04 + 0.005 * lw,
             head_length=0.06 + 0.005 * lw,
             fc=color, ec=color, alpha=alpha, zorder=7)


    
def draw_transition_graph(obj):
    if len(obj.params.time_point_label) > 1:
        time_points = obj.params.time_point_label
    else:
        time_points = [f'T{str(i)}' for i in range(1, obj.tp_data_num+1)]
        
    interval_conv = dict()
    for i in range(len(time_points)-1):
        time_1 = time_points[i]
        time_2 = time_points[i+1]
        interval_conv[i] = f'{time_1} → {time_2}'    

    p_th = 0.05
    df = obj.pval_df.copy()
    df = df[df['p-value'] < p_th]
    intervals = sorted(df["Interval"].unique())

    # === 모든 노드 이름 수집 ===
    all_nodes = sorted(set(df["source"]).union(df["target"]))
    node_color_map = {n: obj.celltype_colors[n] for i, n in enumerate(all_nodes)}
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(intervals), figsize=(5*len(intervals), 5), constrained_layout=True)
    
    
    fig.set_constrained_layout_pads(
        w_pad=0.3,   # subplot 간 가로 여백
        h_pad=0.5,   # subplot 간 세로 여백
        hspace=0.07, # 상하 패널 간 거리 (subplot이 여러 행일 때)
        wspace=0.05   # 좌우 패널 간 거리
    )
    if len(intervals) == 1:
        axes = [axes]

    for i, interval in enumerate(intervals):
        ax = axes[i]
        sub_df = df[df["Interval"] == interval]

        G = nx.DiGraph()
        for _, row in sub_df.iterrows():
            G.add_edge(row["source"], row["target"],
                    weight=-np.log10(row["p-value"]),
                    interval=row["Interval"])
            
        pos = nx.shell_layout(G)

        # --- EDGE DRAW ---
        for (u, v, d) in G.edges(data=True):
            # p-value 기반 시각 강도 조절
            lw = np.clip(0.5 + 0.3 * d["weight"], 0.8, 6.0)     # 두께: -log10(p)
            alpha = np.clip(0.2 + 0.025 * d["weight"], 0.3, 0.9)  # 투명도: -log10(p)

            color = node_color_map.get(u, "#999999")
            rad = 0.05 * (hash(u) % 5 - 2) / 2  # 곡률 랜덤화로 겹침 줄이기

            if u == v:
                continue
            else:
                ax.annotate(
                    "",
                    xy=pos[v], xycoords="data",
                    xytext=pos[u], textcoords="data",
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color,
                        lw=lw,
                        alpha=alpha,
                        shrinkA=10, shrinkB=10,
                        connectionstyle=f"arc3,rad={rad}",
                    ),
                )

        # --- NODE DRAW ---
        for n in G.nodes():
            x, y = pos[n]
            c = node_color_map.get(n, "#999999")
            ax.scatter(x, y, s=1300, color=c, alpha=0.25, zorder=1, edgecolors='none')
            ax.scatter(x, y, s=700, color=c, alpha=0.95, edgecolors="white", linewidth=0.8, zorder=2)

            # === 노드 밑에 cell type 이름 추가 ===
            # n이 이미 cell type name일 경우 그대로 사용
            ct_name = n 
            ax.text(
                x, y - 0.2,                # y 아래로 살짝 내림
                ct_name,
                ha='center', va='top',
                fontsize=10,
                color='black',
                fontweight='medium',
                zorder=3
                )
        
        # --- SELF-LOOP EDGE --- #
        for (u, v, d) in G.edges(data=True):
            if u == v:
                x, y = pos[u]
                color = node_color_map.get(u, "gray")
                angle = (hash(u) % 6) * 60  # 방향 랜덤 분산 (0~300도)
                weight = d["weight"]        # p-value 기반 두께
                draw_true_self_loop(ax, x, y, color=color, radius=0.1, angle=angle, weight=weight)
                
        # --- panel title ---
        ax.set_title(f"{interval_conv[interval]}", fontsize=15, weight="bold", pad=10)
        ax.axis("off")

    # === LEGEND ===
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=n,
            markerfacecolor=node_color_map[n], markersize=8)
        for n in all_nodes
    ]
    # plt.savefig("transition_graph.pdf", dpi=400, bbox_inches='tight', transparent=True)
    plt.show()


# 2025-11-04: Add figure for gene-gene transiiton matrix
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
import matplotlib.colors as mcolors

# === circle을 legend에서 유지하기 위한 핸들러 ===
def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    # facecolor를 안전하게 추출
    fc = orig_handle.get_facecolor()
    if hasattr(fc, "__len__"):
        if len(fc) == 1:
            fc = fc[0]
    fc = mcolors.to_rgba(fc)  # ✅ float 튜플 → RGBA 확정

    return mpatches.Circle((width / 2, height / 2),
                           radius=height / 2.5,
                           facecolor=fc,
                           edgecolor='black',
                           lw=0.4)

def draw_gg_matrix(obj):
    for interval_idx, edge_list in obj.selected_edges.items():
        # ------- Data ------- #
        df = pd.DataFrame(edge_list, columns=["source","target","jaccard","cosine","rank"])
        df[['t1_time', 't1_ct', 't1_gc']] = df['source'].str.extract(r"t(\d+)_(\d+)_(\d+)")
        df[['t2_time', 't2_ct', 't2_gc']] = df['target'].str.extract(r"t(\d+)_(\d+)_(\d+)")
        df[['t1_ct', 't1_gc', 't2_ct', 't2_gc']] = df[['t1_ct', 't1_gc', 't2_ct', 't2_gc']].astype(int)
        df['intensity'] = 1 / (df['rank'] + 1e-6)

        try:
            in_ct = obj.tp_data_dict[interval_idx].obs.drop_duplicates()
            type_to_cluster = dict(zip(in_ct['cluster_label'], in_ct[obj.params.cell_type_label]))
            df['t1_ct_name'] = df['t1_ct'].map(type_to_cluster)
            df['t2_ct_name'] = df['t2_ct'].map(type_to_cluster)
        except Exception:
            df['t1_ct_name'] = "CT" + df['t1_ct'].astype(str)
            df['t2_ct_name'] = "CT" + df['t2_ct'].astype(str)
        
        module_to_cluster = {i: f'G{str(i+1)}' for i in range(10)}
        df['t1_gc_name'] = df['t1_gc'].map(module_to_cluster)
        df['t2_gc_name'] = df['t2_gc'].map(module_to_cluster)

        df['source_label'] = df['t1_ct_name'] + "_" + df['t1_gc_name']
        df['target_label'] = df['t2_ct_name'] + "_" + df['t2_gc_name']

        pivot_df = df.pivot_table(index='source_label', columns='target_label',
                                  values='intensity', aggfunc='mean').fillna(0)
        
        if len(obj.params.time_point_label) > 1:
            time_points = obj.params.time_point_label
        else:
            time_points = [f'T{str(i)}' for i in range(1, obj.tp_data_num+1)]
            
        interval_labels = dict()
        for i in range(len(time_points)-1):
            time_1 = time_points[i]
            time_2 = time_points[i+1]
            interval_labels[i] = f'{time_1} → {time_2}' 

        title = interval_labels[interval_idx] if interval_labels else f"Interval {interval_idx}"

        # === 색상 매핑 ===
        ct_color_map = obj.celltype_colors

        row_ct = [label.split('_')[0] for label in pivot_df.index]
        col_ct = [label.split('_')[0] for label in pivot_df.columns]
        row_gc = [label.split('_')[1] for label in pivot_df.index]
        col_gc = [label.split('_')[1] for label in pivot_df.columns]

        # === Heatmap ===
        n_row, n_col = pivot_df.shape

        # 기본 단위 크기 (한 cell cluster당 크기)
        cell_size = 0.4   # 0.4~0.6 정도가 적당함
        base = 2.0        # 최소 margin 확보용 base 크기

        # 행, 열 개수에 따라 figsize 계산
        fig_w = base + n_col * cell_size
        fig_h = base + n_row * cell_size

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            pivot_df,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Transition strength'}
        )

        # === 축 / 눈금 제거 ===
        ax.axis('off')

        n_row, n_col = pivot_df.shape

        # === 원(circle) 추가: 비율 유지 위해 aspect='equal' ===
        ax.set_aspect('equal')  # ✅ 원 찌그러짐 방지

        # 왼쪽 (row)
        for i, (ct, gc) in enumerate(zip(row_ct, row_gc)):
            y_pos = i + 0.5
            color = ct_color_map.get(ct, "gray")
            circle = plt.Circle((-0.6, y_pos), 0.25, color=color, ec='black', lw=0.4, clip_on=False, alpha=0.6)
            ax.add_patch(circle)
            ax.text(-0.6, y_pos, gc, ha='center', va='center', fontsize=10, color='black')

        ## === 아래쪽(열) circle + G-label ===
        for j, (ct, gc) in enumerate(zip(col_ct, col_gc)):
            x_pos = j + 0.5
            color = ct_color_map.get(ct, "gray")
            circle = plt.Circle((x_pos, n_row + 0.6), 0.25, color=color, ec='black', lw=0.4, clip_on=False, alpha=0.6)
            ax.add_patch(circle)
            ax.text(x_pos, n_row + 0.6, gc, ha='center', va='center', fontsize=10, color='black')

        
        # === 축 label 추가 ===
        ax.text(-1.2, n_row / 2, "Source gene clusters", ha='center', va='center',
                rotation=90, fontsize=10)
        ax.text(n_col / 2, n_row + 1.4, "Target gene clusters", ha='center', va='center',
                fontsize=10)

        # === legend: figure 밖으로 이동 ===
        used_celltypes = sorted(set(row_ct) | set(col_ct))
        filtered_ct_color_map = {ct: obj.celltype_colors[ct] for ct in used_celltypes if ct in obj.celltype_colors}

        patches = [
            mpatches.Circle((0, 0), radius=0.22,
                            facecolor=color,  # 여기서 color는 이미 (r,g,b)
                            label=ct, ec='black', lw=0.4)
            for ct, color in filtered_ct_color_map.items()
        ]


        # figure 밖 오른쪽 배치
        legend_x = .85 + (2.0 / fig_w)

        fig.legend(
            handles=patches,
            title="Cell types",
            loc="upper right",
            bbox_to_anchor=(legend_x, 0.95),
            frameon=False,
            handler_map={mpatches.Circle: HandlerPatch(patch_func=make_legend_circle)},
            handletextpad=0.3,
            handlelength=1.0,
            labelspacing=0.3,
            borderaxespad=0.2,
            fontsize=10
            )

        # === 제목 및 레이아웃 ===
        plt.title(f"{title}", pad=10)
        plt.tight_layout()
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
    


