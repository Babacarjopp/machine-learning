"""
==========================================================
  Malware Classifier — Application Streamlit
  Analyse Statique de Fichiers PE (Windows)
  Random Forest Optimisé | F1 = 0.9849
==========================================================

Lancement :
    streamlit run app_streamlit.py

Dépendances :
    pip install streamlit scikit-learn pandas joblib pefile matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────
st.set_page_config(
    page_title="Malware Classifier PE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown("""
<style>
.main-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #e74c3c, #3498db);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-card {
    background: #1e2130; border-radius: 12px; padding: 1rem;
    border-left: 4px solid #3498db; margin: 0.5rem 0;
}
.malware-badge {
    background: #e74c3c; color: white; padding: 8px 20px;
    border-radius: 20px; font-weight: bold; font-size: 1.1rem;
}
.safe-badge {
    background: #2ecc71; color: white; padding: 8px 20px;
    border-radius: 20px; font-weight: bold; font-size: 1.1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────
# CHARGEMENT DES MODÈLES
# ─────────────────────────────────
MODEL_PATH  = "best_rf_model.pkl"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Modèle introuvable : {MODEL_PATH}")
        st.stop()
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model()

FEATURES = [
    'AddressOfEntryPoint', 'MajorLinkerVersion', 'MajorImageVersion',
    'MajorOperatingSystemVersion', 'DllCharacteristics',
    'SizeOfStackReserve', 'NumberOfSections', 'ResourceSize'
]

# ─────────────────────────────────
# SIDEBAR
# ─────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Python.svg/1200px-Python.svg.png", width=60)
    st.markdown("## 🛡️ Malware Classifier")
    st.markdown("**Analyse Statique de Fichiers PE**")
    st.divider()
    st.markdown("### 📊 Modèle")
    st.success("Random Forest Optimisé")
    st.markdown("""
    | Métrique   | Score   |
    |------------|---------|
    | F1-Score   | 0.9849  |
    | Accuracy   | 0.9910  |
    | AUC-ROC    | 0.9994  |
    | Précision  | 0.9839  |
    | Rappel     | 0.9858  |
    """)
    st.divider()
    st.markdown("### ⚙️ Hyperparamètres")
    st.code("""
max_depth       : None
min_samples_split: 5
n_estimators    : 100
max_features    : sqrt
    """)
    st.divider()
    st.caption("Projet Machine Learning | Classification Malware PE")

# ─────────────────────────────────
# TITRE PRINCIPAL
# ─────────────────────────────────
st.markdown('<div class="main-title">🛡️ Malware Classifier — Analyse PE</div>', 
            unsafe_allow_html=True)
st.markdown("**Classification de fichiers Windows (PE) par apprentissage supervisé**")
st.divider()

# ─────────────────────────────────
# TABS
# ─────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prédiction Manuelle", "📁 Prédiction par Fichier PE", "📊 Tableau de Bord"])

# ══════════════════════════════════
# TAB 1 : PRÉDICTION MANUELLE
# ══════════════════════════════════
with tab1:
    st.markdown("### Entrez les caractéristiques du fichier PE")
    st.info("💡 Ces valeurs correspondent aux métadonnées extraites de l'en-tête PE du fichier exécutable.")

    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs['AddressOfEntryPoint']         = st.number_input("AddressOfEntryPoint",          min_value=0, value=10407, help="Point d'entrée du fichier exécutable")
        inputs['MajorLinkerVersion']           = st.number_input("MajorLinkerVersion",           min_value=0, max_value=255, value=9)
        inputs['MajorImageVersion']            = st.number_input("MajorImageVersion",            min_value=0, value=0)
        inputs['MajorOperatingSystemVersion']  = st.number_input("MajorOperatingSystemVersion",  min_value=0, value=4)

    with col2:
        inputs['DllCharacteristics']           = st.number_input("DllCharacteristics",           min_value=0, value=0)
        inputs['SizeOfStackReserve']           = st.number_input("SizeOfStackReserve",           min_value=0, value=1048576)
        inputs['NumberOfSections']             = st.number_input("NumberOfSections",             min_value=1, max_value=96, value=4)
        inputs['ResourceSize']                 = st.number_input("ResourceSize",                 min_value=0, value=952)

    st.divider()
    
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        x = np.array([[inputs[f] for f in FEATURES]])
        proba = model.predict_proba(x)[0]
        pred  = model.predict(x)[0]
        conf  = max(proba)

        col_r, col_p = st.columns(2)
        with col_r:
            if pred == 0:
                st.markdown('<span class="malware-badge">⚠️ MALWARE DÉTECTÉ</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="safe-badge">✅ FICHIER LÉGITIME</span>', unsafe_allow_html=True)

        with col_p:
            st.metric("Confiance", f"{conf*100:.2f}%")
            st.metric("P(Malware)",  f"{proba[0]*100:.2f}%")
            st.metric("P(Légitime)", f"{proba[1]*100:.2f}%")

        # Jauge de confiance
        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(0, proba[0], color='#e74c3c', label='Malware')
        ax.barh(0, proba[1], left=proba[0], color='#2ecc71', label='Légitime')
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel('Probabilité')
        ax.set_title('Distribution des probabilités')
        ax.legend(loc='upper right')
        ax.axvline(x=0.5, color='white', linestyle='--', linewidth=2)
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════
# TAB 2 : FICHIER PE RÉEL
# ══════════════════════════════════
with tab2:
    st.markdown("### 📁 Upload d'un fichier exécutable (.exe / .dll)")
    st.warning("⚠️ L'analyse est **purement statique** — aucun code n'est exécuté.")
    
    uploaded_file = st.file_uploader("Glissez un fichier .exe ou .dll ici", 
                                      type=["exe", "dll"])
    
    if uploaded_file:
        try:
            import pefile, tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            pe = pefile.PE(tmp_path)
            oh = pe.OPTIONAL_HEADER
            
            # Extraction des features
            extracted = {
                'AddressOfEntryPoint':        oh.AddressOfEntryPoint,
                'MajorLinkerVersion':         oh.MajorLinkerVersion,
                'MajorImageVersion':          oh.MajorImageVersion,
                'MajorOperatingSystemVersion':oh.MajorOperatingSystemVersion,
                'DllCharacteristics':         oh.DllCharacteristics,
                'SizeOfStackReserve':         oh.SizeOfStackReserve,
                'NumberOfSections':           pe.FILE_HEADER.NumberOfSections,
                'ResourceSize': (
                    pe.OPTIONAL_HEADER.DATA_DIRECTORY[2].Size
                    if len(pe.OPTIONAL_HEADER.DATA_DIRECTORY) > 2 else 0
                ),
            }
            
            st.success("✅ Features extraites avec pefile :")
            st.dataframe(pd.DataFrame([extracted]).T.rename(columns={0: 'Valeur'}))
            
            x = np.array([[extracted[f] for f in FEATURES]])
            proba = model.predict_proba(x)[0]
            pred  = model.predict(x)[0]

            st.divider()
            st.markdown(f"**Fichier analysé :** `{uploaded_file.name}`")
            if pred == 0:
                st.error(f"⚠️ MALWARE DÉTECTÉ — Confiance : {proba[0]*100:.1f}%")
            else:
                st.success(f"✅ FICHIER LÉGITIME — Confiance : {proba[1]*100:.1f}%")
            
            os.unlink(tmp_path)
            
        except ImportError:
            st.error("❌ `pefile` non installé. Exécutez : `pip install pefile`")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse : {e}")

# ══════════════════════════════════
# TAB 3 : TABLEAU DE BORD
# ══════════════════════════════════
with tab3:
    st.markdown("### 📊 Tableau de Bord — Performance du Modèle")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 F1-Score",   "0.9849", "+0.0207 vs SVM")
    col2.metric("✅ Accuracy",   "0.9910", "+0.0116 vs KNN")
    col3.metric("📡 AUC-ROC",   "0.9994", "Quasi parfait")
    col4.metric("⚡ Entraîn.",   "4.6s",   "vs 23.3s SVM")

    st.divider()

    # Comparaison des modèles
    st.markdown("#### Comparaison des 3 Modèles")
    comp_data = {
        'Modèle': ['Random Forest 🏆', 'KNN', 'SVM'],
        'Accuracy': [0.9918, 0.9794, 0.9543],
        'Précision': [0.9851, 0.9633, 0.9389],
        'Rappel': [0.9874, 0.9679, 0.9053],
        'F1-Score': [0.9863, 0.9656, 0.9218],
        'Temps (s)': [4.6, 0.1, 23.3],
    }
    df_comp = pd.DataFrame(comp_data).set_index('Modèle')
    
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #1a6e3c; color: white' if v else '' for v in is_max]
    
    st.dataframe(
        df_comp.style.apply(highlight_max, subset=['Accuracy','Précision','Rappel','F1-Score'])
                     .format('{:.4f}', subset=['Accuracy','Précision','Rappel','F1-Score'])
                     .format('{:.1f}s', subset=['Temps (s)']),
        use_container_width=True
    )

    # Graphique feature importance
    st.markdown("#### Importance des Features (Random Forest Optimisé)")
    
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(FEATURES)))
    bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
    ax.set_title('Importance des Features — Random Forest', fontweight='bold')
    ax.set_xlabel('Importance (Gini)')
    for bar, val in zip(bars, feat_df['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Hyperparamètres Optimaux (GridSearchCV)")
    st.json({
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "random_state": 42,
        "F1_score_CV": 0.9832,
        "F1_score_test": 0.9849
    })
