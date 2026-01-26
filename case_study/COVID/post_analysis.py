import numpy as np
import pandas as pd

import scipy.stats as stats

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import friedmanchisquare
from scipy.stats import pearsonr

def l2norm(dat):
    norm = np.sqrt(np.sum(np.square(dat), axis=1))
    norm = np.array(norm).reshape((-1, 1))
    norm = dat / norm
    return norm

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