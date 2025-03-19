import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="QUALIFLEX - Analyse Multicritère",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions de l'algorithme QUALIFLEX
def qualiflex(alternatives, criteria, evaluations, weights, orientations):
    """
    Implémentation de l'algorithme QUALIFLEX pour l'aide à la décision multicritère
    """
    # Vérification des dimensions
    n_alternatives = len(alternatives)
    n_criteria = len(criteria)
    
    if evaluations.shape != (n_alternatives, n_criteria):
        raise ValueError("La matrice d'évaluations doit être de dimension (n_alternatives, n_criteria)")
    
    if len(weights) != n_criteria or abs(sum(weights) - 1) > 1e-10:
        raise ValueError("Les poids doivent être de longueur n_criteria et sommer à 1")
    
    if len(orientations) != n_criteria:
        raise ValueError("Les orientations doivent être de longueur n_criteria")
    
    # Génération de toutes les permutations possibles des alternatives
    all_permutations = list(itertools.permutations(range(n_alternatives)))
    
    # Dictionnaire pour stocker les indices de concordance de chaque permutation
    concordance_indices = {}
    
    # Évaluation de chaque permutation
    for perm in all_permutations:
        # Indices de concordance par critère
        criteria_indices = []
        
        for k in range(n_criteria):
            # Indice de concordance pour le critère k
            concordance_k = 0
            
            # Comparaison de chaque paire d'alternatives dans la permutation
            for i in range(n_alternatives):
                for j in range(i+1, n_alternatives):
                    # Alternatives à comparer
                    alt_i = perm[i]
                    alt_j = perm[j]
                    
                    # Évaluations correspondantes
                    eval_i = evaluations[alt_i, k]
                    eval_j = evaluations[alt_j, k]
                    
                    # Calcul de la concordance selon l'orientation du critère
                    if orientations[k] == 'max':
                        if eval_i > eval_j:
                            concordance_k += 1
                        elif eval_i < eval_j:
                            concordance_k -= 1
                    else:  # 'min'
                        if eval_i < eval_j:
                            concordance_k += 1
                        elif eval_i > eval_j:
                            concordance_k -= 1
            
            # Pondération de l'indice de concordance
            weighted_concordance_k = concordance_k * weights[k]
            criteria_indices.append(weighted_concordance_k)
        
        # Indice de concordance global pour cette permutation
        global_concordance = sum(criteria_indices)
        concordance_indices[perm] = global_concordance
    
    # Identification de la permutation optimale
    best_permutation = max(concordance_indices, key=concordance_indices.get)
    
    # Calcul des résultats détaillés pour la meilleure permutation
    detailed_results = calculate_detailed_results(
        best_permutation, alternatives, criteria, evaluations, 
        weights, orientations, n_alternatives, n_criteria
    )
    
    return best_permutation, concordance_indices, detailed_results

def calculate_detailed_results(best_perm, alternatives, criteria, evaluations, weights, orientations, n_alternatives, n_criteria):
    """
    Calcule les résultats détaillés pour la meilleure permutation
    """
    results = {
        "best_ranking": [alternatives[i] for i in best_perm],
        "criteria_concordance": [],
        "pairwise_comparison": {}
    }
    
    for k in range(n_criteria):
        criterion_results = {
            "criterion": criteria[k],
            "orientation": orientations[k],
            "weight": weights[k],
            "pairwise": []
        }
        
        for i in range(n_alternatives):
            for j in range(i+1, n_alternatives):
                alt_i = best_perm[i]
                alt_j = best_perm[j]
                
                eval_i = evaluations[alt_i, k]
                eval_j = evaluations[alt_j, k]
                
                comparison = {
                    "pair": (alternatives[alt_i], alternatives[alt_j]),
                    "evaluations": (eval_i, eval_j),
                }
                
                if orientations[k] == 'max':
                    if eval_i > eval_j:
                        comparison["result"] = "concordant"
                        comparison["value"] = 1
                    elif eval_i < eval_j:
                        comparison["result"] = "discordant"
                        comparison["value"] = -1
                    else:
                        comparison["result"] = "neutral"
                        comparison["value"] = 0
                else:  # 'min'
                    if eval_i < eval_j:
                        comparison["result"] = "concordant"
                        comparison["value"] = 1
                    elif eval_i > eval_j:
                        comparison["result"] = "discordant"
                        comparison["value"] = -1
                    else:
                        comparison["result"] = "neutral"
                        comparison["value"] = 0
                
                criterion_results["pairwise"].append(comparison)
        
        results["criteria_concordance"].append(criterion_results)
    
    return results

# Fonctions pour l'interface
def create_heatmap(evaluations, alternatives, criteria):
    """
    Crée une heatmap pour visualiser la matrice d'évaluation
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(evaluations, annot=True, cmap="YlGnBu", fmt=".1f",
                xticklabels=criteria, yticklabels=alternatives, ax=ax)
    plt.title("Matrice d'évaluation des alternatives")
    plt.tight_layout()
    return fig

def create_bar_chart(alternatives, concordance_indices):
    """
    Crée un graphique à barres pour visualiser les scores des meilleures permutations
    """
    # Obtenir les 5 meilleures permutations
    sorted_perms = sorted(concordance_indices.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Créer des labels de permutation en utilisant les indices des permutations
    labels = []
    for perm, _ in sorted_perms:
        # Convertir chaque permutation en une chaîne au format "Alt1 > Alt2 > Alt3"
        perm_label = " > ".join([alternatives[idx] for idx in perm])
        labels.append(perm_label)
    
    scores = [score for _, score in sorted_perms]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, scores, color="steelblue")
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title("Top 5 des permutations par indice de concordance")
    plt.ylabel("Indice de concordance")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_radar_chart(detailed_results, criteria, weights):
    """
    Crée un radar chart pour visualiser la contribution des critères
    """
    # Calculer la contribution de chaque critère
    contributions = []
    for criterion_result in detailed_results["criteria_concordance"]:
        concordant_count = sum(1 for pair in criterion_result["pairwise"] if pair["result"] == "concordant")
        discordant_count = sum(1 for pair in criterion_result["pairwise"] if pair["result"] == "discordant")
        total_pairs = len(criterion_result["pairwise"])
        
        # Calculer la contribution nette
        if total_pairs > 0:
            contribution = (concordant_count - discordant_count) / total_pairs
        else:
            contribution = 0
        
        # Pondérer par le poids du critère
        weighted_contribution = contribution * criterion_result["weight"]
        contributions.append(weighted_contribution)
    
    # Créer le radar chart
    angles = np.linspace(0, 2*np.pi, len(criteria), endpoint=False).tolist()
    
    # Fermer le cercle en répétant le premier angle
    angles += angles[:1]
    contributions += contributions[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, contributions, color='steelblue', alpha=0.25)
    ax.plot(angles, contributions, color='steelblue', linewidth=2)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria)
    ax.set_title("Contribution des critères à la solution optimale")
    
    plt.tight_layout()
    return fig

def to_excel_download_link(df):
    """
    Génère un lien pour télécharger le DataFrame en tant que fichier Excel
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Résultats')
    writer.close()
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode('utf-8')
    return f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'

def normalize_weights(weights):
    """
    Normalise les poids pour qu'ils somment à 1
    """
    sum_weights = sum(weights)
    if sum_weights > 0:
        return [w / sum_weights for w in weights]
    return weights

# Interface utilisateur
def main():
    # CSS pour styliser l'application
    st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
        text-align: center;
    }
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .section {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #e9ecef;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .footnote {
        font-size: 12px;
        color: #6c757d;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

    # En-tête
    st.markdown('<div class="main-header">QUALIFLEX - Analyse Multicritère d\'Aide à la Décision</div>', unsafe_allow_html=True)
    
    # Description
    with st.expander("📖 À propos de QUALIFLEX"):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("""
            *QUALIFLEX* (QUALItative FLEXible multiple criteria method) est une méthode d'analyse multicritère 
            développée pour résoudre des problèmes de décision complexes. Cette méthode est particulièrement 
            adaptée aux situations où un décideur doit choisir entre plusieurs alternatives en se basant 
            sur multiples critères potentiellement contradictoires.

            ### Principes fondamentaux
            - *Approche par permutation* : QUALIFLEX examine toutes les permutations possibles des alternatives.
            - *Indices de concordance* : Pour chaque permutation, l'algorithme calcule des indices mesurant la concordance entre le classement et les performances des alternatives.
            - *Pondération* : Les critères peuvent être pondérés selon leur importance relative.
            - *Flexibilité* : La méthode peut traiter des critères quantitatifs ou qualitatifs.

            Cette application vous permet d'expérimenter avec l'algorithme QUALIFLEX en définissant vos propres 
            alternatives, critères, et préférences.
            """)
        with col2:
            st.image("https://via.placeholder.com/300x300.png?text=QUALIFLEX", width=200)

    # Création de la barre latérale
    with st.sidebar:
        st.markdown("### Configuration de l'analyse")
        
        # Choix entre exemple et saisie manuelle
        input_type = st.radio("Sélectionnez une méthode d'entrée des données:", 
                              ["Utiliser l'exemple prédéfini", "Saisir mes propres données"])
        
        if input_type == "Utiliser l'exemple prédéfini":
            # Exemple prédéfini
            n_alternatives = 3
            n_criteria = 4
            alternatives = ["Site A", "Site B", "Site C"]
            criteria = ["Coût", "Proximité", "Impact Env.", "Main-d'œuvre"]
            orientations = ["min", "max", "min", "max"]
            weights = [0.3, 0.2, 0.3, 0.2]
            evaluations = np.array([
                [7, 8, 5, 6],  # Site A
                [5, 6, 4, 8],  # Site B
                [8, 9, 7, 5]   # Site C
            ])
        else:
            # Saisie manuelle
            st.markdown("#### 1. Définir les alternatives et critères")
            n_alternatives = st.slider("Nombre d'alternatives:", min_value=2, max_value=6, value=3)
            n_criteria = st.slider("Nombre de critères:", min_value=2, max_value=8, value=4)
            
            # Saisie des noms des alternatives
            alternatives = []
            for i in range(n_alternatives):
                alt = st.text_input(f"Nom de l'alternative {i+1}:", value=f"Alternative {i+1}")
                alternatives.append(alt)
            
            # Saisie des critères, orientations et poids
            criteria = []
            orientations = []
            weights = []
            
            for i in range(n_criteria):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    crit = st.text_input(f"Critère {i+1}:", value=f"Critère {i+1}")
                    criteria.append(crit)
                with col2:
                    orient = st.selectbox(f"Sens {i+1}:", ["max", "min"], key=f"orient_{i}")
                    orientations.append(orient)
                with col3:
                    weight = st.number_input(f"Poids {i+1}:", min_value=0.0, max_value=1.0, value=1.0/n_criteria, format="%.2f")
                    weights.append(weight)
            
            # Normalisation des poids
            weights = normalize_weights(weights)
            
            # Saisie des évaluations
            st.markdown("#### 2. Matrice d'évaluation")
            evaluations = np.zeros((n_alternatives, n_criteria))
            
            for i in range(n_alternatives):
                st.markdown(f"{alternatives[i]}")
                cols = st.columns(n_criteria)
                for j, col in enumerate(cols):
                    with col:
                        evaluations[i, j] = st.number_input(
                            f"{criteria[j]}",
                            min_value=0.0,
                            max_value=10.0,
                            value=5.0,
                            format="%.1f",
                            key=f"eval_{i}_{j}"
                        )
        
        # Bouton pour lancer l'analyse
        run_analysis = st.button("Lancer l'analyse QUALIFLEX")

    # Corps principal
    if run_analysis:
        with st.spinner("Analyse en cours..."):
            # Création de l'onglet pour les résultats
            tab1, tab2, tab3, tab4 = st.tabs(["Résumé", "Matrice d'évaluation", "Classement détaillé", "Exportation"])
            
            # Exécution de l'algorithme QUALIFLEX
            best_perm, concordance_indices, detailed_results = qualiflex(
                alternatives, criteria, evaluations, weights, orientations
            )
            
            # Création du DataFrame de la matrice d'évaluation
            df_evaluations = pd.DataFrame(evaluations, 
                                        index=alternatives, 
                                        columns=criteria)
            
            # Conversion des orientations en DataFrame
            orientations_df = pd.DataFrame([orientations], columns=criteria, index=["Orientation"])
            orientations_df = orientations_df.replace({"max": "↑ Maximiser", "min": "↓ Minimiser"})
            
            # Conversion des poids en DataFrame
            weights_df = pd.DataFrame([weights], columns=criteria, index=["Poids"])
            
            with tab1:
                st.markdown('<div class="sub-header">Résumé des Résultats</div>', unsafe_allow_html=True)
                
                # Affichage du classement optimal
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.markdown('<div class="section">', unsafe_allow_html=True)
                    st.markdown("### 🏆 Classement optimal des alternatives")
                    
                    # Créer une dataframe pour le classement
                    ranking_df = pd.DataFrame({
                        "Rang": range(1, len(best_perm) + 1),
                        "Alternative": [alternatives[i] for i in best_perm]
                    })
                    
                    # Afficher avec un style
                    st.dataframe(
                        ranking_df,
                        column_config={
                            "Rang": st.column_config.NumberColumn(format="%d"),
                            "Alternative": "Alternative"
                        },
                        hide_index=True,
                        width=300
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="section">', unsafe_allow_html=True)
                    st.markdown("### 📊 Statistiques globales")
                    
                    # Nombre total de permutations
                    total_perms = len(concordance_indices)
                    
                    # Indice de concordance de la meilleure permutation
                    best_concordance = concordance_indices[best_perm]
                    
                    st.metric("Permutations analysées", f"{total_perms}")
                    st.metric("Indice de concordance optimal", f"{best_concordance:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top permutations
                    st.markdown('<div class="section">', unsafe_allow_html=True)
                    st.markdown("### Top 5 des meilleures permutations")
                    # Passage d'alternatives au lieu de best_ranking 
                    bar_fig = create_bar_chart(alternatives, concordance_indices)
                    st.pyplot(bar_fig)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                 #   st.markdown('<div class="section">', unsafe_allow_html=True)
                  #  st.markdown("### Contribution des critères")
                   # radar_fig = create_radar_chart(detailed_results, criteria, weights)
                    #st.pyplot(radar_fig)
                    #st.markdown('</div>', unsafe_allow_html=True)
                    st.dataframe(
                        pd.concat([df_evaluations, orientations_df, weights_df]),
                        height=300
                    )
                
            with tab2:
                st.markdown('<div class="sub-header">Matrice d\'évaluation</div>', unsafe_allow_html=True)
                
                # Affichage de la matrice d'évaluation
                st.markdown('<div class="section">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.dataframe(
                        pd.concat([df_evaluations, orientations_df, weights_df]),
                        height=300
                    )
                
                with col2:
                    heatmap_fig = create_heatmap(evaluations, alternatives, criteria)
                    st.pyplot(heatmap_fig)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="sub-header">Analyse Détaillée</div>', unsafe_allow_html=True)
                
                # Affichage des résultats détaillés par critère
                for idx, criterion_result in enumerate(detailed_results["criteria_concordance"]):
                    criterion = criterion_result["criterion"]
                    orientation = criterion_result["orientation"]
                    weight = criterion_result["weight"]
                    
                    st.markdown(f'<div class="section">', unsafe_allow_html=True)
                    st.markdown(f"### Critère: {criterion}")
                    st.markdown(f"*Orientation:* {'Maximiser' if orientation == 'max' else 'Minimiser'} | *Poids:* {weight:.2f}")
                    
                    # Calcul des statistiques pour ce critère
                    concordant_count = sum(1 for pair in criterion_result["pairwise"] if pair["result"] == "concordant")
                    discordant_count = sum(1 for pair in criterion_result["pairwise"] if pair["result"] == "discordant")
                    neutral_count = sum(1 for pair in criterion_result["pairwise"] if pair["result"] == "neutral")
                    
                    total_pairs = len(criterion_result["pairwise"])
                    concordance_rate = concordant_count / total_pairs * 100 if total_pairs > 0 else 0
                    
                    # Affichage des métriques
                    cols = st.columns(3)
                    cols[0].metric("Paires concordantes", f"{concordant_count}/{total_pairs}", f"{concordance_rate:.1f}%")
                    cols[1].metric("Paires discordantes", f"{discordant_count}/{total_pairs}")
                    cols[2].metric("Paires neutres", f"{neutral_count}/{total_pairs}")
                    
                    # Tableau des comparaisons par paires
                    st.markdown("#### Comparaisons par paires")
                    
                    pairs_data = []
                    for comp in criterion_result["pairwise"]:
                        pair = comp["pair"]
                        evals = comp["evaluations"]
                        result = comp["result"]
                        value = comp["value"]
                        
                        pairs_data.append({
                            "Alternative 1": pair[0],
                            "Évaluation 1": evals[0],
                            "Alternative 2": pair[1],
                            "Évaluation 2": evals[1],
                            "Résultat": result,
                            "Valeur": value
                        })
                    
                    pairs_df = pd.DataFrame(pairs_data)
                    
                    # Remplacer les résultats par des emoji
                    pairs_df["Résultat"] = pairs_df["Résultat"].replace({
                        "concordant": "✅ Concordant",
                        "discordant": "❌ Discordant",
                        "neutral": "➖ Neutre"
                    })
                    
                    st.dataframe(pairs_df, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown('<div class="sub-header">Exportation des Résultats</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="section">', unsafe_allow_html=True)
                # Création d'un dataframe pour l'exportation
                export_data = {
                    "Alternatives": alternatives,
                }
                
                for j, criterion in enumerate(criteria):
                    export_data[f"{criterion} ({orientations[j]})"] = [evaluations[i, j] for i in range(n_alternatives)]
                
                export_df = pd.DataFrame(export_data)
                
                # Ajout du classement
                ranking_dict = {alternatives[i]: idx+1 for idx, i in enumerate(best_perm)}
                export_df["Classement"] = export_df["Alternatives"].map(ranking_dict)
                
                # Affichage du dataframe
                st.dataframe(export_df)
                
                # Lien de téléchargement
                excel_link = to_excel_download_link(export_df)
                st.markdown(f"### Télécharger les résultats")
                st.markdown(
                    f'<a href="{excel_link}" download="QUALIFLEX_Results.xlsx" '
                    f'style="text-decoration:none;'
                    f'display:inline-block;'
                    f'background-color:#1E3A8A;'
                    f'color:white;'
                    f'padding:10px 20px;'
                    f'border-radius:5px;'
                    f'text-align:center;'
                    f'width:100%;">'
                    f'📥 Télécharger les résultats en Excel</a>', 
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Affichage par défaut lorsque l'analyse n'a pas encore été lancée
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown("## Bienvenue dans l'application QUALIFLEX")
        st.markdown("""
        Cette application vous permet d'analyser des problèmes de décision multicritère en utilisant la méthode QUALIFLEX.
        
        ### Comment utiliser cette application:
        
        1. *Configurez votre analyse* dans le panneau latéral:
           - Choisissez d'utiliser l'exemple prédéfini ou saisissez vos propres données
           - Définissez vos alternatives et critères
           - Spécifiez les orientations (maximiser ou minimiser) et les poids des critères
           - Entrez les évaluations pour chaque alternative selon chaque critère
        
        2. *Lancez l'analyse* en cliquant sur le bouton "Lancer l'analyse QUALIFLEX"
        
        3. *Explorez les résultats* dans les différents onglets:
           - Résumé: vue d'ensemble des résultats principaux
           - Matrice d'évaluation: visualisation des données d'entrée
           - Classement détaillé: analyse approfondie des concordances par critère
           - Exportation: téléchargement des résultats en format Excel
        
        Cliquez sur "📖 À propos de QUALIFLEX" en haut de la page pour en savoir plus sur cette méthode d'analyse multicritère.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Pied de page
    st.markdown('<div class="footnote">© 2025 - Application développée avec Streamlit pour présenter l\'algorithme QUALIFLEX</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
