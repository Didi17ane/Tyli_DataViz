import streamlit as st
import pandas as pd
import numpy as np
import json
from Regle_Segmentation import generate_cluster_profiles, assign_cluster_from_rules

# Variables utilisées pour la segmentation
SEGMENTATION_VARS = [
    "bancarise", "sex", "marital_status",
    "milieu_resid", "region_name", "city", "age_group"
]

def safe_sort_clusters(cluster_list):
    """Trie les clusters de manière robuste (gère int et str)"""
    try:
        # Essayer de convertir tous en int
        return sorted(cluster_list, key=lambda x: int(x) if str(x).isdigit() else float('inf'))
    except (ValueError, TypeError):
        # Sinon, trier comme strings
        return sorted(cluster_list, key=str)

# --- START: Diagnostic / explication des règles (définir AVANT usage) ---
def _normalize_value(v):
    if pd.isna(v):
        return ""
    return str(v).strip().upper()

def _normalize_expected(vals):
    if vals is None:
        return set()
    if isinstance(vals, list):
        return set(_normalize_value(v) for v in vals if pd.notna(v))
    return {_normalize_value(vals)}

def _row_rule_detail(row, conds):
    per_var = {}
    total = 0
    matched = 0
    for var, expected in conds.items():
        total += 1
        if var not in row.index:
            per_var[var] = False
            continue
        val = _normalize_value(row[var])
        expected_set = _normalize_expected(expected)
        ok = (val in expected_set)
        per_var[var] = ok
        if ok:
            matched += 1
    pct = matched / total if total > 0 else 0.0
    return {"per_var": per_var, "total": total, "matched": matched, "pct": pct}

def explain_rules_for_dataframe(df_rows, rules, top_k_candidates=3, sample_limit=200):
    rows = df_rows.copy().reset_index()
    if len(rows) > sample_limit:
        rows = rows.head(sample_limit)

    out = []
    formatted = {}
    for cid, rule in rules.items():
        if isinstance(rule, list):
            conds = {}
            for r in rule:
                if isinstance(r, dict):
                    for k, v in r.items():
                        conds[k] = v
        elif isinstance(rule, dict):
            conds = dict(rule)
        else:
            conds = {}
        formatted[str(cid)] = conds

    for _, row in rows.iterrows():
        candidates = []
        for cid, conds in formatted.items():
            detail = _row_rule_detail(row, conds)
            if detail["pct"] > 0:
                candidates.append((cid, detail["pct"], detail["matched"], detail["total"], detail["per_var"]))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        top = candidates[:top_k_candidates]
        out.append({
            "orig_index": row["index"],
            "best_candidates": top,
            "n_candidates": len(candidates),
            "row_preview": {c: row.get(c, "") for c in ["age_num", "sex", "marital_status", "city", "milieu_resid", "region_name", "bancarise"] if c in row.index}
        })
    return pd.DataFrame(out)
# --- END: Diagnostic / explication des règles ---

st.set_page_config(page_title="Analyse des Clusters", layout="wide")
st.title("🧬 Plateforme d'Analyse et Segmentation par Clusters")


# ==========================================================
# SECTION 0 : ANALYSE MILIEU DE RÉSIDENCE
# ==========================================================
st.header("🗺️ Analyse Milieu de Résidence par Sous-Prefecture")

uploaded_analyse = st.file_uploader(
    "📌 Importer un dataset pour analyser milieu_resid vs region", 
    type="csv",
    key="upload_analyse"
)

if uploaded_analyse:
    df_analyse = pd.read_csv(uploaded_analyse)
    
    if "city" in df_analyse.columns and "milieu_resid" in df_analyse.columns:
        
        # Tableau croisé
        st.subheader("📊 Tableau croisé : Sous-Préfecture × Milieu de résidence")
        
        cross_tab = pd.crosstab(
            df_analyse["city"], 
            df_analyse["milieu_resid"],
            margins=True,
            margins_name="Total"
        )
        
        st.dataframe(cross_tab, use_container_width=True)
        
        # Pourcentages par région
        st.subheader("📈 Pourcentages par Sous-Préfecture")
        
        cross_pct = pd.crosstab(
            df_analyse["city"], 
            df_analyse["milieu_resid"],
            normalize='index'
        ) * 100
        
        cross_pct = cross_pct.round(2)
        st.dataframe(cross_pct.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        csv_cross = cross_pct.to_csv(index=True).encode("utf-8")

        st.download_button(
            label="📥 Télécharger les pourcentages (CSV)",
            data=csv_cross,
            file_name="pourcentages_sous_prefecture.csv",
            mime="text/csv",
            key="download_cross_pct"
        )
        # Visualisation graphique
        st.subheader("📊 Visualisation graphique")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Répartition par sous-préfecture (effectifs)**")
            cross_tab_plot = cross_tab.drop('Total', errors='ignore').drop('Total', axis=1, errors='ignore')
            st.bar_chart(cross_tab_plot)
        
        with col2:
            st.write("**Répartition par sous-préfecture (pourcentages)**")
            st.bar_chart(cross_pct)
        
        # Identification des régions mono-milieu
        st.subheader("🔍 Identification des sous-préfecture à milieu unique")
        
        Spréfecture_info = []
        for sp in df_analyse["city"].unique():
            sub = df_analyse[df_analyse["city"] == sp]
            milieux = sub["milieu_resid"].unique()
            
            if len(milieux) == 1:
                Spréfecture_info.append({
                    "Sous-Préfecture": sp,
                    "Milieu unique": milieux[0],
                    "Effectif": len(sub),
                    "Type": "🔴 Mono-milieu"
                })
            else:
                pct = sub["milieu_resid"].value_counts(normalize=True) * 100
                dominant = pct.idxmax()
                pct_dominant = pct.max()
                
                Spréfecture_info.append({
                    "Sous-Préfecture": sp,
                    "Milieu dominant": f"{dominant} ({pct_dominant:.1f}%)",
                    "Effectif": len(sub),
                    "Type": "🟢 Multi-milieu"
                })
        
        df_sp = pd.DataFrame(Spréfecture_info)
        st.dataframe(df_sp, use_container_width=True)
        
        # Téléchargement de l'analyse
        csv_analyse = df_sp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger l'analyse des régions",
            data=csv_analyse,
            file_name="analyse_SPrefecture_milieu.csv",
            mime="text/csv"
        )
        
        st.divider()
    else:
        st.error("❌ Le dataset doit contenir les colonnes 'city' et 'milieu_resid'")


# ==========================================================
# SECTION 1 : UPLOAD ET GÉNÉRATION DES PROFILS
# ==========================================================
st.header("1️⃣ Importer le dataset avec clusters")

uploaded = st.file_uploader("📌 Choisir un fichier CSV contenant une colonne 'cluster'", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    st.subheader("📄 Aperçu du dataset chargé")
    st.dataframe(df.head(10))
    
    st.info(f"**Dimensions :** {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    # Vérifier la présence de la colonne cluster
    if "cluster" not in df.columns:
        st.error("❌ Le dataset doit contenir une colonne 'cluster'")
        st.stop()
    
    #st.success(f"✅ {df['cluster'].nunique()} clusters détectés : {sorted(df['cluster'].unique())}")
   
    # Forcer le type du cluster initial en string pour la suite
    df["cluster"] = df["cluster"].astype("string")
    
    # ==========================================================
    # GÉNÉRATION DES PROFILS
    # ==========================================================
    st.header("2️⃣ Génération des profils et règles")
    
    threshold = st.slider(
        "Seuil de définition des modalités (%)", 
        min_value=50, 
        max_value=95, 
        value=80, 
        step=5,
        help="Pourcentage cumulé pour identifier les modalités clés d'un cluster"
    ) / 100
    
    if st.button("🔍 Générer les profils et règles de segmentation"):
        with st.spinner("Génération en cours..."):
            profiles, summaries, rules, df_modalities = generate_cluster_profiles(df, threshold=threshold)
        
        st.success("🎉 Profils et règles générés avec succès !")
        
        # Sauvegarder les règles dans la session
        st.session_state["profiles"] = profiles
        st.session_state["summaries"] = summaries
        st.session_state["rules"] = rules
        st.session_state["df_modalities"] = df_modalities
        
        # Sauvegarder en JSON avec types cohérents
        rules_jsonable = {}
        for k, v in rules.items():
            cluster_key = int(k) if isinstance(k, (int, float, np.integer)) else str(k)
            rules_jsonable[cluster_key] = {
                str(var): [str(val) for val in vals] 
                for var, vals in v.items()
            }
        
        with open("./Rules Clusters/cluster_rules.json", "w", encoding="utf-8") as f:
            json.dump(rules_jsonable, f, ensure_ascii=False, indent=4)
        
        st.info("💾 Règles sauvegardées dans `cluster_rules.json`")

    # ==========================================================
    # EXPLORATION DES CLUSTERS
    # ==========================================================
    if "profiles" in st.session_state:
        st.header("3️⃣ Explorer les clusters")
        
        profiles = st.session_state["profiles"]
        summaries = st.session_state["summaries"]
        rules = st.session_state["rules"]
        
        # Trier les clusters de manière robuste
        cluster_ids = safe_sort_clusters(list(profiles.keys()))
        
        selected = st.selectbox("📊 Sélectionner un cluster à explorer", cluster_ids)
        
        # Affichage du résumé
        #st.subheader(f"📝 Résumé du Cluster {selected}")
        #st.info(summaries[selected])
        
        # Colonnes pour organisation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Variables numériques")
            if not profiles[selected]["numeric"].empty:
                st.dataframe(
                    profiles[selected]["numeric"][["mean", "50%", "std", "min", "max"]]
                    .rename(columns={"50%": "median"}),
                    use_container_width=True
                )
            else:
                st.write("Aucune variable numérique")
        
        with col2:
            st.subheader("🧩 Règles de segmentation")
            st.json(rules[selected])
        
        # Variables catégorielles détaillées
        st.subheader("📁 Variables catégorielles (utilisées pour la segmentation)")
        
        for var in SEGMENTATION_VARS:
            if var in profiles[selected]["categorical"]:
                with st.expander(f"🔹 {var}"):
                    tab = profiles[selected]["categorical"][var]
                    
                    # Afficher uniquement les modalités définissantes en priorité
                    defining = tab[tab["is_defining"]]
                    other = tab[~tab["is_defining"]]
                    
                    st.write("**Modalités définissantes :**")
                    st.dataframe(
                        defining[["cluster_pct", "global_pct", "cum_pct"]].round(2),
                        use_container_width=True
                    )
                    
                    if not other.empty:
                        st.write("**Autres modalités :**")
                        st.dataframe(
                            other[["cluster_pct", "global_pct"]].round(2),
                            use_container_width=True
                        )

# ==========================================================
# SECTION 2 : SEGMENTATION D'UN NOUVEAU DATASET
# ==========================================================
st.header("4️⃣ Appliquer la segmentation à un nouveau dataset")

uploaded2 = st.file_uploader(
    "📌 Importer un dataset à segmenter (doit contenir les mêmes variables)", 
    type="csv",
    key="upload_new"
)

if uploaded2:
    df_new = pd.read_csv(uploaded2)
    
    st.subheader("📄 Aperçu du nouveau dataset")
    st.dataframe(df_new.head(10))
    
    st.info(f"**Dimensions :** {df_new.shape[0]} lignes × {df_new.shape[1]} colonnes")
    
    # Charger les règles
    try:
        print("Attempting to load rules from JSON...")
        with open("./Rules Clusters/cluster_rules_manual_example.json", "r", encoding="utf-8") as f:
            rules_loaded = json.load(f)
        print("Rules loaded successfully.")
        # Reconvertir au format attendu (forcer tous les clusters en int si possible)
        formatted_rules = {}
        for cluster_key, rule_dict in rules_loaded.items():
            # Essayer de convertir en int, sinon garder comme string
            try:
                print(f"Converting cluster key: {cluster_key}")
                cluster_id = int(float(cluster_key))
            except (ValueError, TypeError):
                cluster_id = str(cluster_key)
                print(f"Keeping cluster key as string: {cluster_id}")
            print(f"Processing rules for cluster {cluster_id}: {rule_dict}")
            
            cond_dict = {}
            for rule in rule_dict:
                for var, vals in rule.items():
                    cond_dict[var] = vals

            formatted_rules[cluster_id] = cond_dict

            #formatted_rules[cluster_id] = {var: vals for var, vals in rule_dict.items()}

            print("OKIIIIIIIIIIIIIIIIIIIIIIIII")
        print("Formatted rules successfully.")
        print(f"Formatted rules: {formatted_rules}")
        st.info(f"✅ Règles chargées pour {len(formatted_rules)} clusters")
        
        if st.button("🔮 Prédire les clusters"):
            with st.spinner("Segmentation en cours..."):
                result = assign_cluster_from_rules(df_new, formatted_rules)
            
            # Harmoniser les types pour éviter les problèmes Arrow
            result["cluster_assigned"] = result["cluster_assigned"].astype("string")
            if "age_group" in result.columns:
                result["age_group"] = result["age_group"].astype("string")
            
            # Sauvegarde dans la session pour réutilisation sans recalcul
            st.session_state["result"] = result
            st.session_state["df_new"] = df_new
            if uploaded and "cluster" in df.columns:
                st.session_state["df_initial"] = df.copy()

            st.session_state["segmentation_done"] = True
            st.success("✅ Segmentation terminée !")
            
        # ==========================================================
        # SECTION AFFICHAGE / ANALYSE APRES SEGMENTATION
        # ==========================================================
        if st.session_state.get("segmentation_done", False):

            result = st.session_state["result"]

            # Statistiques de segmentation
            st.subheader("📊 Résultats de la segmentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total d'individus", len(result))
                st.metric("Individus non assignés", (result["cluster_assigned"] == "Aucun").sum())
            
            with col2:
                assigned = result[result["cluster_assigned"] != "Aucun"]
                if len(assigned) > 0:
                    st.metric("Taux d'assignation", f"{len(assigned)/len(result)*100:.1f}%")
                    st.metric("Score moyen de correspondance", f"{assigned['match_score'].mean():.2f}")
        
            # Distribution par cluster
            st.subheader("📈 Distribution par cluster")
            dist = result["cluster_assigned"].value_counts()
            
            # Réorganiser pour avoir un ordre logique
            dist_sorted = dist.reindex(safe_sort_clusters(dist.index.tolist()), fill_value=0)
            
            st.bar_chart(dist_sorted)
            
            # --- INSERT DEBUG DIAGNOSTIC START ---
            # Affiche diagnostic d'application des règles (non-assignés + vérif assignés)
            # Récupère les règles formatées (peut provenir de la variable locale ou de la session)
            rules_for_debug = formatted_rules if 'formatted_rules' in locals() else st.session_state.get("rules", {})
            if rules_for_debug:
                non_assignes = result[result["cluster_assigned"] == "Aucun"].copy()
                assignes = result[result["cluster_assigned"] != "Aucun"].copy()

                st.subheader("🔎 Diagnostic application des règles (aperçu)")

                # Paramètres d'échantillon
                sample_limit = st.number_input("Nombre de lignes à analyser (preview)", min_value=10, max_value=30000, value=200, step=10, key="dbg_sample")
                top_k = st.number_input("Top clusters candidats à afficher par ligne", min_value=1, max_value=10, value=3, key="dbg_topk")

                # Non-assignés
                if len(non_assignes) > 0:
                    st.markdown("### ❌ Non-assignés (échantillon)")
                    # diagnostic sample (affichage)
                    df_dbg_na = explain_rules_for_dataframe(non_assignes, rules_for_debug, top_k_candidates=top_k, sample_limit=sample_limit)
                    # formater les candidats pour affichage lisible
                    def fmt_cands(cands):
                        return "; ".join([f"{cid} ({pct*100:.0f}%, {m}/{t})" for cid, pct, m, t, per in cands]) if cands else ""
                    df_dbg_na["candidates_txt"] = df_dbg_na["best_candidates"].map(fmt_cands)
                    st.dataframe(df_dbg_na[["orig_index", "n_candidates", "candidates_txt", "row_preview"]].head(200), use_container_width=True)
                    st.download_button("📥 Télécharger diagnostic non-assignés (JSON)", data=df_dbg_na.to_json(orient="records", force_ascii=False).encode("utf-8"), file_name="debug_non_assignes.json", mime="application/json")
                    
                    # Options pour appliquer des réassignations aux non-assignés
                    st.divider()
                    st.write("### 🔁 Réassigner les non-assignés")
                    method = st.radio(
                        "Méthode de réassignation",
                        options=[
                            "Top-candidat (diagnostic) — utiliser le meilleur candidat trouvé",
                            "Réappliquer les règles d'origine (assign_cluster_from_rules)"
                        ],
                        index=0
                    )
                    pct_threshold = st.slider("Seuil minimal (%) pour accepter une assignation", min_value=0, max_value=100, value=0, step=5, key="reassign_pct")
                    
                    # Optionnel : calculer les candidats pour l'ensemble des non-assignés (pas seulement l'échantillon)
                    if st.checkbox("Calculer candidats pour tous les non-assignés (peut être lent)", value=False):
                        df_dbg_na_full = explain_rules_for_dataframe(non_assignes, rules_for_debug, top_k_candidates=max(1, top_k), sample_limit=len(non_assignes))
                    else:
                        df_dbg_na_full = df_dbg_na.copy()
                    
                    if st.button("🛠️ Appliquer la réassignation aux non-assignés"):
                        updated = st.session_state.get("result", result).copy()
                        assigned_count = 0
                        
                        if method.startswith("Top-candidat"):
                            # Utiliser le top-candidat extrait du diagnostic
                            for _, r in df_dbg_na_full.iterrows():
                                orig_idx = r["orig_index"]
                                best = r["best_candidates"]
                                if best and len(best) > 0:
                                    top_cid, top_pct, matched, total, per_var = best[0]
                                    if (top_pct * 100) >= pct_threshold:
                                        if orig_idx in updated.index:
                                            updated.at[orig_idx, "cluster_assigned"] = str(top_cid)
                                            updated.at[orig_idx, "match_score"] = float(top_pct)
                                            updated.at[orig_idx, "assigned_via"] = "top_candidate"
                                            assigned_count += 1
                        else:
                            # Réappliquer les règles d'origine sur l'ensemble des non-assignés
                            try:
                                non_ass_df = non_assignes.copy()
                                assigned_by_rules = assign_cluster_from_rules(non_ass_df, rules_for_debug)
                                # parcourir et appliquer si score >= seuil et cluster != 'Aucun'
                                for idx, row_ass in assigned_by_rules.iterrows():
                                    cid = row_ass.get("cluster_assigned", "Aucun")
                                    score = row_ass.get("match_score", 0.0)
                                    if cid is not None and str(cid) != "Aucun" and (float(score) * 100) >= pct_threshold:
                                        if idx in updated.index:
                                            updated.at[idx, "cluster_assigned"] = str(cid)
                                            updated.at[idx, "match_score"] = float(score)
                                            updated.at[idx, "assigned_via"] = "reapplied_rules"
                                            assigned_count += 1
                            except Exception as e:
                                st.error(f"Erreur lors de la réapplication des règles : {e}")
                        
                        # Sauvegarder et propager le changement pour la suite du code
                        st.session_state["result"] = updated
                        result = st.session_state["result"]
                        st.success(f"✅ {assigned_count} individus réassignés (seuil {pct_threshold}%)")
                        
                        # Mettre à jour les variables locales utilisées plus bas
                        non_assignes = result[result["cluster_assigned"] == "Aucun"].copy()
                        assignes = result[result["cluster_assigned"] != "Aucun"].copy()
                        
                        # Rafraîchir l'affichage immédiat des diagnostics si nécessaire
                        st.experimental_rerun()
                else:
                    st.info("Aucun non-assigné à diagnostiquer.")

                # Assignés (vérification rapide)
                if len(assignes) > 0:
                    st.markdown("### ✅ Assignés (vérification rapide)")
                    df_dbg_a = explain_rules_for_dataframe(assignes, rules_for_debug, top_k_candidates=top_k, sample_limit=min(sample_limit, 200))
                    df_dbg_a["candidates_txt"] = df_dbg_a["best_candidates"].map(fmt_cands)
                    st.dataframe(df_dbg_a[["orig_index", "n_candidates", "candidates_txt", "row_preview"]].head(200), use_container_width=True)
                    st.download_button("📥 Télécharger diagnostic assignés (JSON)", data=df_dbg_a.to_json(orient="records", force_ascii=False).encode("utf-8"), file_name="debug_assignes.json", mime="application/json")
                else:
                    st.info("Aucun assigné à vérifier.")
            else:
                st.info("Aucune règle disponible pour le diagnostic (générez ou chargez les règles).")
            # --- INSERT DEBUG DIAGNOSTIC END ---
            
            # Préparer le dataset final avec colonnes sélectionnées
            display_cols = ["cluster_assigned"]
            
            # Ajouter les colonnes dans l'ordre souhaité si elles existent
            desired_order = ["age_num", "sex", "marital_status", "city", "milieu_resid", "region_name", "bancarise"]
            for col in desired_order:
                if col in result.columns:
                    display_cols.append(col)
            
            # Ajouter les autres colonnes restantes
            for col in result.columns:
                if col not in display_cols and col not in ["match_score", "age_group"]:
                    display_cols.append(col)
            
            # Renommer cluster_assigned en cluster pour l'affichage
            result_display = result[display_cols].copy()
            result_display.rename(columns={"cluster_assigned": "cluster"}, inplace=True)
            
            # Affichage du résultat
            st.subheader("🗂️ Dataset segmenté")
            st.dataframe(result_display, use_container_width=True)
            
            # Téléchargement
            csv = result_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les résultats (CSV)",
                data=csv,
                file_name="Segmentation_Results.csv",
                mime="text/csv"
            )
            print("______________________________________azerty__________________________")
            print(f"df_new : {df_new}")
            print(f"df_new columns : {df_new.columns}")
            print(f"CSV : {result_display}")
            
            #____________________________________________________________________________

            # ==========================================================
            # ANALYSE CROISÉE : CLUSTERS INITIAUX VS PRÉDITS
            # ==========================================================
            if uploaded and "cluster" in df.columns:
                print("______________________________________zerty__________________________")
                st.divider()
                st.header("🔍 Analyse croisée : Clusters initiaux vs Prédits")
                
                st.info("Cette analyse compare les clusters d'origine avec ceux prédits par les règles de segmentation.")
                
                # Préparer les données pour l'analyse
                df_comparison = result.copy()
                df_comparison["cluster_initial"] = df["cluster"].astype("string")
                df_comparison["cluster_predit"] = df_comparison["cluster_assigned"].astype("string")

                if "Modalité" in df_comparison.columns:
                    df_comparison["Modalité"] = df_comparison["Modalité"].astype("string")
                
                # Indicateur de concordance
                df_comparison["concordance"] = df_comparison["cluster_initial"] == df_comparison["cluster_predit"]

                if "Modalité" in df_comparison.columns:
                    df_comparison["Modalité"] = df_comparison["Modalité"].astype("string")
                
                st.session_state["df_comparison"] = df_comparison.copy()

                
                # =================== STATISTIQUES GLOBALES ===================
                st.subheader("📊 Statistiques globales")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(df_comparison)
                concordants = df_comparison["concordance"].sum()
                non_concordants = total - concordants
                taux_concordance = (concordants / total) * 100
                
                with col1:
                    st.metric("Total individus", total)
                with col2:
                    st.metric("✅ Concordants", concordants, delta=f"{taux_concordance:.1f}%")
                with col3:
                    st.metric("❌ Non concordants", non_concordants, delta=f"{100-taux_concordance:.1f}%", delta_color="inverse")
                with col4:
                    non_assignes = (df_comparison["cluster_predit"] == "Aucun").sum()
                    st.metric("⚠️ Non assignés", non_assignes)
                
                # =================== MATRICE DE CONFUSION ===================
                st.subheader("📋 Matrice de confusion")
                
                confusion_matrix = pd.crosstab(
                    df_comparison["cluster_initial"],
                    df_comparison["cluster_predit"],
                    rownames=["Cluster Initial"],
                    colnames=["Cluster Prédit"],
                    margins=True,
                    margins_name="Total"
                )
                
                confusion_matrix = confusion_matrix.astype("int64")
                st.dataframe(confusion_matrix, use_container_width=True)
                
                # Taux de concordance par cluster initial
                st.subheader("📈 Taux de concordance par cluster initial")
                
                concordance_by_cluster = []
                for cluster in sorted(df_comparison["cluster_initial"].unique(), key=str):
                    sub = df_comparison[df_comparison["cluster_initial"] == cluster]
                    total_cluster = len(sub)
                    concordant = sub["concordance"].sum()
                    non_concordant = total_cluster - concordant
                    non_assigne = (sub["cluster_predit"] == "Aucun").sum()
                    taux = (concordant / total_cluster) * 100 if total_cluster > 0 else 0
                    
                    concordance_by_cluster.append({
                        "Cluster": cluster,
                        "Total": total_cluster,
                        "✅ Concordants": concordant,
                        "❌ Non concordants": non_concordant,
                        "⚠️ Non assignés": non_assigne,
                        "Taux concordance (%)": round(taux, 1)
                    })
                
                df_concordance = pd.DataFrame(concordance_by_cluster)
                
                # Graphique de concordance
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(df_concordance, use_container_width=True, hide_index=True)
                
                with col2:
                    chart_data = df_concordance.set_index("Cluster")["Taux concordance (%)"]
                    st.bar_chart(chart_data)
                    
                # Analyse des erreurs par cluster
                st.subheader("🔎 Détail des erreurs par cluster initial")
                
                cluster_options = safe_sort_clusters(df_comparison["cluster_initial"].astype(str).unique().tolist())
                
                cluster_to_analyze = st.selectbox(
                    "Sélectionner un cluster à analyser en détail",
                    options=cluster_options
                )
                
                # Filtrer les erreurs pour ce cluster
                errors = df_comparison[
                    (df_comparison["cluster_initial"].astype(str) == str(cluster_to_analyze)) & 
                    (~df_comparison["concordance"])
                ]
                
                # Comparer avec les individus correctement classés
                correct = df_comparison[
                    (df_comparison["cluster_initial"].astype(str) == str(cluster_to_analyze)) & 
                    (df_comparison["concordance"])
                ]
                
                if len(errors) > 0:
                    st.warning(f"⚠️ {len(errors)} individus mal classés pour le cluster {cluster_to_analyze}")
                    
                    # Distribution des prédictions erronées
                    st.write(f"**Où sont-ils prédits ?**")
                    error_dist = errors["cluster_predit"].value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(error_dist.rename("Effectif"), use_container_width=True)
                    with col2:
                        st.bar_chart(error_dist)
                    
                    # Analyse des variables pour comprendre les erreurs
                    st.write(f"**Analyse des différences par variable**")
                    
                    # Variables de segmentation à analyser
                    vars_to_check = [v for v in SEGMENTATION_VARS if v in errors.columns]
                    
                    differences = []
                    
                    
                    for var in vars_to_check:
                        # Distribution dans les erreurs
                        error_dist_var = errors[var].value_counts(normalize=True) * 100
                        # Distribution dans les corrects
                        correct_dist_var = correct[var].value_counts(normalize=True) * 100
                        
                        # Identifier les modalités problématiques
                        for modality in error_dist_var.index:
                            pct_error = error_dist_var.get(modality, 0)
                            pct_correct = correct_dist_var.get(modality, 0)
                            diff = pct_error - pct_correct
                            
                            if abs(diff) > 5:  # Différence significative
                                differences.append({
                                    "Variable": var,
                                    "Modalité": modality,
                                    "% dans erreurs": round(pct_error, 1),
                                    "% dans corrects": round(pct_correct, 1),
                                    "Différence": round(diff, 1),
                                    "Impact": "🔴 Sur-représenté" if diff > 0 else "🟢 Sous-représenté"
                                })
                    
                    if differences:
                        df_diff = pd.DataFrame(differences).sort_values("Différence", key=abs, ascending=False)
                       
                        # >>> FIX Arrow : forcer Modalité en string
                        if "Modalité" in df_diff.columns:
                            df_diff["Modalité"] = df_diff["Modalité"].astype("string")
                        # <<<
                        st.dataframe(df_diff, use_container_width=True, hide_index=True)
                        
                        st.info("""
                        **💡 Interprétation :**
                        - **🔴 Sur-représenté** : Cette modalité apparaît plus souvent dans les erreurs → Peut-être ajouter/modifier une règle
                        - **🟢 Sous-représenté** : Cette modalité apparaît moins dans les erreurs → La règle fonctionne bien pour elle
                        """)
                    else:
                        st.success("Pas de différence significative détectée sur les variables de segmentation")
                    
                    # Afficher quelques exemples d'erreurs
                    st.write(f"**📋 Exemples d'individus mal classés (max 20)**")
                    
                    cols_to_show = ["cluster_initial", "cluster_predit"]
                    if "match_score" in errors.columns:
                        cols_to_show.append("match_score")
                    cols_to_show += vars_to_check
                    cols_to_show = [c for c in cols_to_show if c in errors.columns]
                    
                    st.dataframe(errors[cols_to_show].head(20), use_container_width=True)
                    
                    # Export des erreurs
                    csv_errors = errors.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"📥 Télécharger toutes les erreurs du cluster {cluster_to_analyze}",
                        data=csv_errors,
                        file_name=f"erreurs_cluster_{cluster_to_analyze}.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.success(f"✅ Tous les individus du cluster {cluster_to_analyze} sont correctement classés !")
                
                # Export de l'analyse complète
                st.divider()
                st.subheader("📥 Exports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export matrice de confusion
                    csv_confusion = confusion_matrix.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger la matrice de confusion",
                        data=csv_confusion,
                        file_name="matrice_confusion.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export de tous les non-concordants
                    all_errors = df_comparison[~df_comparison["concordance"]]
                    csv_all_errors = all_errors.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger tous les non-concordants",
                        data=csv_all_errors,
                        file_name="tous_non_concordants.csv",
                        mime="text/csv"
                    )
            #____________________________________________________________________________
            
    except FileNotFoundError:
        st.error("⚠️ Aucune règle trouvée. Veuillez d'abord générer les profils dans la section précédente.")
    except Exception as e:
        st.error(f"❌ Erreur lors de la segmentation : {str(e)}")