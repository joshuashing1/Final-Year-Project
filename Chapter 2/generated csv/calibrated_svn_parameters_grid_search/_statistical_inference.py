import numpy as np
import pandas as pd

datasets = [
    {"csv_path": r"Chapter 2\parametric_models\calibrated_svn_parameters_grid_search\USD_svn_grid_search_betas.csv", "title": "USD"}, 
    {"csv_path": r"Chapter 2\parametric_models\calibrated_svn_parameters_grid_search\CNY_svn_grid_search_betas.csv", "title": "CNY"},
    {"csv_path": r"Chapter 2\parametric_models\calibrated_svn_parameters_grid_search\GBP_svn_grid_search_betas.csv", "title": "GBP"}, 
    {"csv_path": r"Chapter 2\parametric_models\calibrated_svn_parameters_grid_search\SGD_svn_grid_search_betas.csv", "title": "SGD"}, 
    {"csv_path": r"Chapter 2\parametric_models\calibrated_svn_parameters_grid_search\EUR_svn_grid_search_betas.csv", "title": "EUR"} 
]

def compute_stats(df: pd.DataFrame):
    out = {}

    # betas: direct quartiles
    for b in ["beta1", "beta2", "beta3", "beta4"]:
        q1 = np.percentile(df[b].values, 25)
        q3 = np.percentile(df[b].values, 75)
        out[b] = (q1, q3)

    # lambdas: log-quartiles
    for l in ["lambd1", "lambd2"]:
        vals = df[l].values
        vals = vals[vals > 0]
        logv = np.log(vals)
        q1 = np.exp(np.percentile(logv, 25))
        q3 = np.exp(np.percentile(logv, 75))
        out[l] = (q1, q3)

    return out

rows = []

for d in datasets:
    df = pd.read_csv(d["csv_path"])
    stats = compute_stats(df)
    print(f"\n{d['title']} :")
    for k, (q1, q3) in stats.items():
        print(f"{k:7s} | Q1: {q1:10.6f} | Q3: {q3:10.6f}")
        rows.append({
            "Currency": d["title"],
            "Parameter": k,
            "Q1": q1,
            "Q3": q3
        })

stats_df = pd.DataFrame(rows)
stats_df.to_csv("svn_grid_search_betas_quartiles.csv", index=False)

print("\nSaved statistics to svn_grid_search_betas_quartiles.csv")
