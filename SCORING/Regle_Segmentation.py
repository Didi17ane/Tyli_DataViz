import pandas as pd
import numpy as np
import json

def create_age_bins():
    """Crée les tranches d'âge standardisées"""
    bins = [0, 14, 24, 34, 44, 54, 64, 74, 84, 94, np.inf]
    labels = [
        "0-14", "15-24", "25-34", "35-44", "45-54",
        "55-64", "65-74", "75-84", "85-94", "95+"
    ]
    return bins, labels


def generate_cluster_profiles(df, threshold=0.8):
    """
    Génère les profils complets des clusters avec règles de segmentation
    
    Retour :
    - profiles : dict cluster -> {numeric: DataFrame, categorical: dict(var->table)}
    - summaries : dict cluster -> texte résumé
    - rules : dict cluster -> dict(var -> list(defining_modalities))
    - df_modalities_all : DataFrame détaillé des modalités
    """
    df = df.copy()
    
    # Créer les tranches d'âge si age_num existe
    bins, labels = create_age_bins()
    if "age_num" in df.columns:
        df["age_group"] = pd.cut(df["age_num"], bins=bins, labels=labels, right=True)
    else:
        df["age_group"] = pd.NA

    # Détection des variables catégorielles
    categorical_vars = df.select_dtypes(include="object").columns.tolist()
    forced_categorical = [
        "bancarise", "sex", "marital_status", 
        "milieu_resid", "region_name", "city", "age_group"
    ]
    
    for col in forced_categorical:
        if col in df.columns and col not in categorical_vars:
            categorical_vars.append(col)
    
    # Retirer 'cluster' des variables à analyser
    categorical_vars = [c for c in categorical_vars if c != "cluster"]

    # Variables numériques
    numeric_vars = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_vars = [n for n in numeric_vars if n != "cluster"]

    profiles = {}
    summaries = {}
    rules = {}
    modal_rows = []

    clusters = sorted(df["cluster"].unique())

    for c in clusters:
        sub = df[df["cluster"] == c]
        cluster_size = len(sub)
        
        # Statistiques numériques
        num_stats = sub[numeric_vars].describe().T if numeric_vars else pd.DataFrame()

        cat_results = {}

        # Analyse de chaque variable catégorielle
        for var in categorical_vars:
            if var not in df.columns:
                continue

            # Fréquences dans le cluster et globalement
            freq_cluster = sub[var].value_counts(normalize=True).rename("cluster_pct") * 100
            freq_global = df[var].value_counts(normalize=True).rename("global_pct") * 100

            tab = pd.concat([freq_cluster, freq_global], axis=1).fillna(0)
            tab = tab.sort_values(by="cluster_pct", ascending=False)
            
            # Calcul cumulatif
            tab["cum_pct"] = tab["cluster_pct"].cumsum()
            tab["surrepresented"] = tab["cluster_pct"] >= (threshold * 100)

            # Identifier les modalités définissantes (cumulant >= threshold%)
            defining = []
            cum = 0.0
            for idx, rowr in tab.iterrows():
                defining.append(idx)
                cum += rowr["cluster_pct"]
                if cum >= (threshold * 100):
                    break

            tab["is_defining"] = tab.index.isin(defining)

            cat_results[var] = tab
            
            # Stocker pour export détaillé
            for modality, rowr in tab.iterrows():
                modal_rows.append({
                    "cluster": c,
                    "variable": var,
                    "modality": modality,
                    "cluster_pct": float(rowr["cluster_pct"]),
                    "global_pct": float(rowr["global_pct"]),
                    "cum_pct": float(rowr["cum_pct"]),
                    "surrepresented_bool": bool(rowr["surrepresented"]),
                    "is_defining": bool(rowr["is_defining"])
                })

        # Résumé automatique du cluster
        summary_parts = []
        for var, tab in cat_results.items():
            defs = tab[tab["is_defining"]].index.astype(str).tolist()
            if defs:
                summary_parts.append(f"{var}: {', '.join(defs)}")
            elif not tab.empty:
                summary_parts.append(f"{var}: {tab.index[0]}")

        summary_text = f"Cluster {c} (n={cluster_size}) → " + " | ".join(summary_parts)

        # Règles : dictionnaire var -> liste des modalités définissantes
        rule_dict = {}
        for var, tab in cat_results.items():
            defining_mods = tab[tab["is_defining"]].index.astype(str).tolist()
            if defining_mods:
                rule_dict[var] = defining_mods

        profiles[c] = {"numeric": num_stats, "categorical": cat_results}
        summaries[c] = summary_text
        rules[c] = rule_dict

    # DataFrame complet des modalités
    df_modalities_all = pd.DataFrame(modal_rows)

    return profiles, summaries, rules, df_modalities_all


# def assign_cluster_from_rules(df_new, rules):
#     """
#     Assigne un cluster à chaque ligne selon les règles extraites
    
#     rules = dict :
#         {
#             0: {"sex": ["Féminin"], "marital_status": ["Marié(e)"], ...},
#             1: {...}
#         }
#     """
#     df_result = df_new.copy()
    
#     # Créer age_group si nécessaire
#     if "age_num" in df_result.columns and "age_group" not in df_result.columns:
#         bins, labels = create_age_bins()
#         df_result["age_group"] = pd.cut(df_result["age_num"], bins=bins, labels=labels, right=True)
    
#     df_result["cluster_assigned"] = "Aucun"
#     df_result["match_score"] = 0

#     for cluster_id, conditions in rules.items():
#         mask = pd.Series(True, index=df_result.index)
#         score = 0

#         for var, allowed_values in conditions.items():
#             if var not in df_result.columns:
#                 mask &= False
#                 continue
            
#             var_mask = df_result[var].astype(str).isin([str(v) for v in allowed_values])
#             mask &= var_mask

#         # Compter le nombre de conditions respectées
#         for idx in df_result[mask].index:
#             matched = sum(
#                 1 for var, vals in conditions.items()
#                 if var in df_result.columns and str(df_result.loc[idx, var]) in [str(v) for v in vals]
#             )
            
#             # Assigner si meilleur score
#             if matched > df_result.loc[idx, "match_score"]:
#                 df_result.loc[idx, "cluster_assigned"] = cluster_id
#                 df_result.loc[idx, "match_score"] = matched

#     return df_result



def assign_cluster_from_rules(df_new, rules):
    """
    Assigne un cluster à chaque ligne selon les règles extraites

    rules = dict :
        0: {"sex": ["Féminin"], "marital_status": ["Marié(e)"], ...},
        1: {...}
    """
    df_result = df_new.copy()

    # Créer age_group si nécessaire
    if "age_num" in df_result.columns and "age_group" not in df_result.columns:
        bins, labels = create_age_bins()
        df_result["age_group"] = pd.cut(
            df_result["age_num"],
            bins=bins,
            labels=labels,
            right=True
        )

    df_result["cluster_assigned"] = "Aucun"
    df_result["match_score"] = 0

    for cluster_id, conditions in rules.items():
        # AND entre champs
        mask = pd.Series(True, index=df_result.index)

        for var, allowed_values in conditions.items():
            if var not in df_result.columns:
                mask &= False
                continue

            mask &= df_result[var].astype(str).isin([str(v) for v in allowed_values])

        # Tous les individus qui respectent TOUTES les conditions de ce cluster
        matched_idx = df_result[mask].index

        # Score = nb de variables du cluster (toutes matchent puisqu'on est dans mask)
        score = len(conditions)

        # On n’écrase que si ce cluster matche plus de conditions que le précédent
        better = df_result.loc[matched_idx, "match_score"] < score
        idx_update = matched_idx[better]

        df_result.loc[idx_update, "cluster_assigned"] = str(cluster_id)
        df_result.loc[idx_update, "match_score"] = score

    return df_result

# -------------------------
# Exemple d'utilisation
# -------------------------
if __name__ == "__main__":
    df = pd.read_csv("../DATAS/ANSTAT2021_clusters_PC.csv")
    profiles, summaries, rules, df_modalities = generate_cluster_profiles(df, threshold=0.8)

    # Sauvegardes
    pd.Series(summaries).rename("summary").to_csv("./Rules Clusters/cluster_summaries.csv", index=True)
    df_modalities.to_csv("./Rules Clusters/cluster_modalities.csv", index=False)

    # Règles en JSON
    rules_jsonable = {
        str(k): {str(var): list(map(str, vals)) for var, vals in v.items()}
        for k, v in rules.items()
    }
    
    with open("./Rules Clusters/cluster_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_jsonable, f, ensure_ascii=False, indent=4)

    # Profils numériques
    rows = []
    for cl, prof in profiles.items():
        if not prof["numeric"].empty:
            row = prof["numeric"][["mean", "50%", "std"]].rename(columns={"50%": "median"}).reset_index()
            row["cluster"] = cl
            rows.append(row)
    
    if rows:
        df_profiles_num = pd.concat(rows, ignore_index=True)
        df_profiles_num.to_csv("./Rules Clusters/cluster_profiles_numeric.csv", index=False)

    print("✅ Exports terminés : cluster_summaries.csv, cluster_modalities.csv, cluster_rules.json, cluster_profiles_numeric.csv")

    