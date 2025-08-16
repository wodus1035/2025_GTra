
import numpy as np
import pandas as pd
import gseapy as gp
import seaborn as sns
import networkx as nx

import itertools

import sys
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colorbar as colorbar
import plotly.graph_objects as go

import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import friedmanchisquare
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("/data1/home/jyj/PROJECT/2025_GTra/src")

import GTra as gt

from gutils import l2norm, JS
from pathlib import Path


def plot_paper_transition(adata, obj):
    celltype_label = obj.params.cell_type_label
    n_colors = adata.obs[celltype_label].nunique()

    cb_palette = sns.color_palette("Set2", n_colors).as_hex()
    color_mapping = {key: cb_palette[idx] for idx, key in enumerate(adata.obs[celltype_label].unique())}

    if len(obj.params.time_point_label) > 1:
        time_points = obj.params.time_point_label
    else:
        time_points = [f"T{str(i)}" for i in range(1, obj.tp_data_num+1)]
    
    interval_conv = dict()
    for i in range(len(time_points)-1):
        t1, t2 = time_points[i], time_points[i+1]
        interval_conv[i] = f'{t1}→{t2}'
    
    df = obj.pval_df.copy()
    fname = obj.params.answer_path_dir
    if os.path.isfile(fname):
        obj.answer_path = pd.read_csv(fname, sep=",")
        answer_edges = set(zip(obj.answer_path["source"],obj.answer_path["target"]))
        filtered_df = df[df.apply(
            lambda row: (row["source"], row["target"]) in answer_edges, axis=1)]
        df = filtered_df.copy()

    threshold = 0.05
    unique_intervals = df["Interval"].unique()
    interval_graphs = []

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
    # all_celltypes = list(all_node_names)

    norm = plt.Normalize(vmin=all_weights.min(), vmax=all_weights.max())
    cmap_edges = plt.cm.Reds

    fig_output = f"{obj.params.output_dir}/figures"
    os.makedirs(fig_output, exist_ok=True)

    for interval, G in interval_graphs:
        fig, ax = plt.subplots(figsize=(5, 5))
        node_colors = [color_mapping.get(n, "#cccccc") for n in G.nodes()]

        pos = nx.shell_layout(G)
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        edge_colors = cmap_edges(norm(weights))

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                            edgecolors="black", node_size=1200, ax=ax, linewidths=2)

        for node, (x, y) in pos.items():
            txt = ax.text(x - 0.2, y - 0.2, node, fontsize=15, color='dimgray', style='italic')

        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            edge_cmap=cmap_edges,
            edge_vmin=all_weights.min(),
            edge_vmax=all_weights.max(),
            arrowstyle="->",
            arrowsize=15,
            width=2.0,
            alpha=0.9,
            ax=ax,
            connectionstyle="arc3,rad=0.3",
            min_target_margin=15
        )

        ax.set_title(f"{interval_conv[interval]}", fontsize=20, weight="bold")
        ax.set_facecolor("#f7f7f7")
        ax.axis("off")
        # break

        # 저장
        out_path_svg = f"{fig_output}/{obj.params.output_name}_interval_{interval}.svg"
        out_path_pdf = f"{fig_output}/{obj.params.output_name}_interval_{interval}.pdf"
        plt.tight_layout()
        plt.savefig(out_path_svg)
        plt.savefig(out_path_pdf)
        plt.show()
        plt.close(fig)  # 메모리 절약

        # 예: 기존 그래프에서 사용한 weight 범위
    vmin = min(all_weights)     # 최소 -log10(p) 값
    vmax = max(all_weights)     # 최대 -log10(p) 값 (원하는 범위로 조절 가능)

    # 정규화 객체와 컬러맵 지정
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.Reds

    # figure + colorbar 전용 axis
    fig, ax = plt.subplots(figsize=(1.2, 4))
    fig.subplots_adjust(right=0.5)

    # colorbar 생성용 dummy ScalarMappable
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # 필수: 빈 array 지정

    # colorbar 그리기
    cbar = plt.colorbar(sm, cax=ax)
    cbar.set_label(r'$-\log_{10}$(p-value)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    
    plt.savefig(f"{fig_output}/{obj.params.output_name}_colorbar.svg")
    plt.savefig(f"{fig_output}/{obj.params.output_name}_colorbar.pdf")
    plt.close()

def time_ct_convert(gtra, time):
    celltype_label = gtra.params.cell_type_label
    df = gtra.tp_data_dict[time].obs.copy()
    label_name = "cluster_label"
    ctc = df.drop_duplicates(label_name).set_index(label_name)[celltype_label].to_dict()
    return ctc

def get_path_genes(gtra):
    path_gene_dict = dict()
    for trj, paths in gtra.merge_path_dict.items():
        for path in paths:
            for node_idx in range(len(path)-1):
                src = path[node_idx]
                tar = path[node_idx+1]
                
                st, sc, sg = src.split('_')
                tt, tc, tg = tar.split('_')

                stime = int(st[1])
                ttime = int(tt[1])

                sc, sg = int(sc), int(sg)
                tc, tg = int(tc), int(tg)

                sgenes = gtra.gene_label_info[stime][sc][sg]
                tgenes = gtra.gene_label_info[ttime][tc][tg]
                sr = src[:src.rfind('_')]
                ta = tar[:tar.rfind('_')]
                key = f'{sr}|{ta}'
                inter_genes = list(set(sgenes).intersection(set(tgenes)))
                
                if path_gene_dict.get(key) is None:
                    path_gene_dict[key] = inter_genes
                else:
                    path_gene_dict[key].extend(inter_genes)
    return path_gene_dict

def get_path_gn(gtra):
    path_gene_dict = get_path_genes(gtra)
    path_gn = []
    for i, j in path_gene_dict.items():
        src, tar = i.split('|')
        stok = src.split('_')
        ttok = tar.split('_')

        st, sc = int(stok[0][1]), int(stok[1])
        tt, tc = int(ttok[0][1]), int(ttok[1])
        
        scc = time_ct_convert(gtra, st)
        tcc = time_ct_convert(gtra, tt)
        
        stt = scc[sc]
        ttt = tcc[tc]
        
        path_gn.append([st, stt, ttt, len(set(j))])
    path_gn = pd.DataFrame(path_gn, columns=["Interval", "source","target", "GN"])
    return path_gn

def extract_time_and_celltype(label):
    t, cell = label.split(": ")
    return int(t[1:]), cell

def plot_paper_trj(adata, obj):
    celltype_label = obj.params.cell_type_label
    n_colors = adata.obs[celltype_label].nunique()

    cb_palette = sns.color_palette("Set2", n_colors).as_hex()
    color_mapping = {key:cb_palette[idx] for idx,key in enumerate(adata.obs[celltype_label].unique())}

    path_gn = get_path_gn(obj)
    df = path_gn.copy()
    df["Interval"] = df["Interval"].astype(int)

    interval_to_time = {i: f"T{i}" for i in sorted(df["Interval"].unique())}
    next_time = {f"T{i}": f"T{i+1}" for i in sorted(df["Interval"].unique())}

    df["source_label"] = df.apply(lambda x: f"{interval_to_time[x['Interval']]}: {x['source']}", axis=1)
    df["target_label"] = df.apply(lambda x: f"{next_time[interval_to_time[x['Interval']]]}: {x['target']}", axis=1)

    labels = sorted(set(df["source_label"]).union(set(df["target_label"])),
                    key=lambda x: extract_time_and_celltype(x))
    
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    node_colors = [color_mapping[label.split(": ")[1]] for label in labels]

    time_labels = sorted(set(l.split(":")[0] for l in labels))
    desired_celltype_order = adata.obs[celltype_label].unique()

    cell_types_per_time = {
        t: [ct for ct in desired_celltype_order if ct in set(l.split(": ")[1] for l in labels if l.startswith(t))]
        for t in time_labels
    }


    spacing_factor = 1.1

    node_x = []
    node_y = []
    for label in labels:
        t, ct = label.split(": ")
        ct_list = cell_types_per_time[t]
        x = time_labels.index(t) / (len(time_labels) - 1) if len(time_labels) > 1 else 0.5
        y_rank = ct_list.index(ct)
        y = (y_rank / max(1, len(ct_list) - 1)) * spacing_factor
        node_x.append(x)
        node_y.append(1 - y)


    df["source_index"] = df["source_label"].map(label_to_index)
    df["target_index"] = df["target_label"].map(label_to_index)
    link_colors = [node_colors[idx] for idx in df["source_index"]]

    visible_labels = []
    for label in labels:
        t, _ = label.split(": ")
        visible_labels.append(label.split(": ")[1])


    fig = go.Figure(data=[go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=15,
            thickness=10,
            line=dict(color="black", width=0.8),
            # label=labels,
            label=visible_labels,  # T0만 이름 표시

            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=df["source_index"],
            target=df["target_index"],
            # value=-np.log10(df["GN"]),
            value=df["GN"],
            color=link_colors  # 🔥 수정된 부분
        )
    )])

    fig.update_layout(
        font_size=12,
        height=400,  # 👈 y축 늘렸다면 여기 높이도 늘려야 함
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='white',  # 또는 투명하게 'rgba(0,0,0,0)'
        plot_bgcolor='white',
        # margin=dict(l=0, r=0, t=0, b=0)
    )

    tieme_anno = obj.params.time_point_label
    for i, t in enumerate(tieme_anno):
        x = i / (len(tieme_anno) - 1) if len(tieme_anno) > 1 else 0.5
        fig.add_annotation(
            x=x, y=-0.32,  # y는 다이어그램 하단에 위치하도록 음수로 조정
            text=t,
            showarrow=False,
            xref="paper", yref="paper",
            font=dict(size=14),
            align="center"
        )
        
    fig_output = f"{obj.params.output_dir}/figures"
    output_svg = f"{fig_output}/{obj.params.output_name}_trj.svg"
    output_pdf = f"{fig_output}/{obj.params.output_name}_trj.pdf"
    
    fig.write_image(output_svg, format="svg", height=300)
    fig.write_image(output_pdf, format="pdf", height=300)
    # fig.show()

def detect_expression_trend(expression_vector):
    """
    expression_vector: list or array of average expression across timepoints
    """
    x = np.array(expression_vector)
    t = np.arange(len(x))
    
    # Normalize for stability
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-6)

    # Check increasing/decreasing using correlation with time
    corr, _ = pearsonr(x_norm, t)
    
    if corr > 0.8:
        return "increasing"
    elif corr < -0.8:
        return "decreasing"
    # elif np.argmax(x) in range(1, len(x)-1) and np.argmin(x) not in [0, len(x)-1]:
    #     return "mid-peak"
    elif np.std(x) < 0.05:
        return "flat"
    else:
        return "transient"

def select_sig_pattern(obj):
    pt = pd.read_csv(f'{obj.params.output_dir}/{obj.params.output_name}_pattern_genes.csv',index_col=0)
    pattern_data = obj.merge_pattern_dict.copy()
    pt_key_dict = {i[i.find('[')+1:i.find(']')]:i[:i.find('[')] for i in pt.columns}
    sig_pt_dat = {i: pattern_data[i] for i in pt_key_dict.keys()}
    
    pattern_stats = []
    for pattern_id, df in sig_pt_dat.items():
        # 시간점별 그룹화
        df = l2norm(df)
        avg_expr  = df.values.mean(axis=0)
        trend = detect_expression_trend(avg_expr)
        genes = ';'.join(list(df.index))

        time_cols = [col for col in df.columns if col.startswith("t")]
        time_series = [df[col] for col in time_cols]
        f_stat, p_value = friedmanchisquare(*time_series)

        # t0, t1, t2, t3 = df['t0'], df['t1'], df['t2'], df['t3']
        # f_stat, p_value = friedmanchisquare(t0, t1, t2, t3)
        log_p = -np.log10(p_value) if p_value > 0 else np.inf
        trj = pt_key_dict[pattern_id]
        pattern_stats.append({
            "Pattern_ID": pattern_id,
            "Trend":trend,
            "F_statistic": f_stat,
            "p_value": p_value,
            "-log10(p_value)":log_p,
            "trajectory":trj,
            "nGenes":len(df.index),
            "Genes":genes,
        })

    result_df = pd.DataFrame(pattern_stats)
    result_df = result_df[result_df["p_value"]<1e-2].sort_values('p_value')
    return result_df

def create_pattern_dist_matrix(obj, result_df):
    # result_df = select_sig_pattern(obj)
    pt_dist = []
    for i in result_df['Pattern_ID']:
        expr1 = l2norm(obj.merge_pattern_dict[i])
        avg_expr1 = expr1.mean(axis=0).values
        tmp=[]
        for j in result_df['Pattern_ID']:
            expr2 = l2norm(obj.merge_pattern_dict[j])
            avg_expr2 = expr2.mean(axis=0).values
            
            # dist = cosine_similarity([avg_expr1], [avg_expr2])[0][0]
            rho, p_val = stats.pearsonr(avg_expr1, avg_expr2)
            # print(p_val)
            tmp.append(rho)
        pt_dist.append(tmp)

    pt_dist = pd.DataFrame(pt_dist, index=result_df["trajectory"], columns=result_df["trajectory"])
    return pt_dist

def intra_pattern_clustering(obj, K=3):
    result_df = select_sig_pattern(obj)
    pt_dist = create_pattern_dist_matrix(obj, result_df)
    
    linked = linkage(pt_dist, "average")
    clusters = fcluster(linked, K, criterion="maxclust")

    pt_df = result_df[["Pattern_ID", "trajectory"]].copy()
    pt_df["cluster"] = clusters

    return pt_df, result_df, pt_dist

def convert_to_icons(traj_str, celltype_to_icon):
    steps = traj_str.split("->")
    return "→".join([celltype_to_icon.get(step.strip(), "⬛") for step in steps])

def plot_paper_intra_pattern(adata, obj, K=3):
    pt_df, result_df, pt_dist = intra_pattern_clustering(obj, K=K)
    clusters = pt_df["cluster"].values.tolist()

    celltype_label = obj.params.cell_type_label
    n_colors = adata.obs[celltype_label].nunique()

    cb_palette = sns.color_palette("Set2", n_colors).as_hex()
    celltype_to_icon = {ct: str(idx+1)for idx, ct in enumerate(adata.obs[celltype_label].unique())}
    yticklabels_iconified = [convert_to_icons(traj,celltype_to_icon) for traj in pt_dist.index]


    c_lut = {c+1:cb_palette[c] for c in range(K)}
    trend = pd.Series(clusters, index=result_df.index)
    col_colors = trend.map(c_lut)

    sns.set(style="whitegrid", context="notebook", font_scale=.8)

    g = sns.clustermap(
        pt_dist, 
        col_colors=col_colors.to_numpy(),
        xticklabels=False,
        yticklabels=False,
        # yticklabels=yticklabels_iconified,
        # linewidths=0.7,
        cbar_pos=(0.05, 0.8, 0.02, 0.18),
        cmap='vlag', 
        method="average",
        figsize=(6, 6)
        )
    g.ax_heatmap.set_xlabel("")    # x축 제목
    g.ax_heatmap.set_ylabel("")    # y축 제목
    g.cax.set_ylabel("Correlation", rotation=270, labelpad=10)

    cluster_legend = [mpatches.Patch(color=c_lut[i], label=f'C{i}')
                    for i in sorted(np.unique(pt_df["cluster"].values))]

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

    # color_row = col_colors_raw  # shape: (N,)
    color_row = np.array(col_colors_raw)
    col_order = g.dendrogram_col.reordered_ind
    ordered_colors = color_row[col_order]

    boundaries = []
    for color, group in itertools.groupby(enumerate(ordered_colors), key=lambda x: tuple(x[1])):
        group = list(group)
        start = group[0][0]
        end = group[-1][0]
        boundaries.append((start, end, color))

    ax = g.ax_heatmap
    edge_color='white'
    face_alpha=0.07
    for start, end, color in boundaries:
        size = end - start + 1
        rect = mpatches.Rectangle(
            (start, start), size, size,
            linewidth=3.5,
            edgecolor=edge_color,
            facecolor=color + (face_alpha,),
            linestyle='-',
            joinstyle='round'
        )    
        ax.add_patch(rect)
    

    fig_output = f"{obj.params.output_dir}/figures"
    plt.savefig(f"{fig_output}/{obj.params.output_name}_heat.svg", bbox_inches="tight")
    plt.savefig(f"{fig_output}/{obj.params.output_name}_heat.pdf", bbox_inches="tight")
    plt.show()
    plt.close()

def plot_rep_patterns(obj,adata, K=3):
    celltype_label = obj.params.cell_type_label
    n_colors = adata.obs[celltype_label].nunique()

    cb_palette = sns.color_palette("Set2", n_colors).as_hex()

    sns.set(style="white", context="paper")
    pt_df, result_df, _ = intra_pattern_clustering(obj, K=K)
    n_clusters = pt_df["cluster"].nunique()
    fig, axes = plt.subplots(n_clusters, 1, figsize=(4, 2.5 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]

    pt_id_c = dict()
    cluster_genes = pd.DataFrame()
    for ax, (c, d) in zip(axes, pt_df.groupby("cluster")):
        tmp = []
        expr_list = []
        union_genes = set()

        for pi in d["Pattern_ID"]:
            tmp.append(pi)
            expr = l2norm(obj.merge_pattern_dict[pi])
            avg_expr = expr.mean(axis=0)
            expr_list.append(avg_expr)

            # gene set: expr의 index가 gene 이름이라면 그대로 사용
            union_genes.update(list(expr.index))


        gene_df = pd.DataFrame(list([union_genes])).T
        gene_df.columns = [[f'C{c}']]

        if c == 1:
            cluster_genes = gene_df
        else:
            cluster_genes = pd.concat([cluster_genes, gene_df],axis=1)

        pt_id_c[c] = tmp

        expr_df = pd.concat(expr_list, axis=1)
        mean_expr = expr_df.mean(axis=1)
        std_expr = expr_df.std(axis=1)

        # 평균선
        ax.plot(obj.params.time_point_label, mean_expr.values,
                label=None,
                marker='o',
                color=cb_palette[c-1],
                linewidth=2)

        # 표준편차 shading
        ax.fill_between(obj.params.time_point_label,
                        mean_expr.values - std_expr.values,
                        mean_expr.values + std_expr.values,
                        color=cb_palette[c-1],
                        alpha=0.2, linewidth=0)

        # 타이틀 (gene 수 포함)
        ax.set_title(f"C{c} (n={len(union_genes)})", fontsize=13, fontweight='bold')

        ax.set_ylabel("Expression", fontsize=11)
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=12)  # 원하는 크기로 변경

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel("Time points", fontsize=11)

    result_df.to_csv(f"{obj.params.output_dir}/{obj.params.output_name}_trjdf.csv")
    pt_df.to_csv(f"{obj.params.output_dir}/{obj.params.output_name}_ptdf.csv")
    cluster_genes.to_csv(f"{obj.params.output_dir}/{obj.params.output_name}_cgenes.csv")

    fig_output = f"{obj.params.output_dir}/figures"
    plt.tight_layout()
    plt.savefig(f"{fig_output}/{obj.params.output_name}_rep_patterns.svg", bbox_inches="tight")
    plt.savefig(f"{fig_output}/{obj.params.output_name}_rep_patterns.pdf", bbox_inches="tight")
    plt.show()
