import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Config générale de la page
st.set_page_config(
    page_title="Spotify Master", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Petit hack CSS pour forcer le mode sombre et styliser les boutons
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    p, h1, h2, h3, li, .stMarkdown { color: #FFFFFF !important; }
    div[data-testid="stMetricValue"] { color: #1DB954 !important; }
    div[data-testid="stMetricLabel"] { color: #b3b3b3 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #181818; border-radius: 10px; padding: 15px; border: 1px solid #282828;
    }
    .stButton>button {
        background-color: #1DB954; color: white; border-radius: 20px; border: none; font-weight: bold;
    }
    .stButton>button:hover { background-color: #1ed760; }
    /* Style pour l'info box */
    div[data-testid="stExpander"] { background-color: #181818; border: 1px solid #282828; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Chargement et nettoyage des données
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('archive/dataset.csv')
    except FileNotFoundError:
        st.error("⚠️ Fichier 'archive/dataset.csv' introuvable.")
        return pd.DataFrame()

    # On vire les doublons sinon ça fausse les résultats
    df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Création d'une colonne combinée pour la recherche
    df['search_label'] = df['track_name'].astype(str) + " - " + df['artists'].astype(str)
    
    # Ajustement des poids pour que ces critères comptent plus dans le calcul
    df['speechiness'] = df['speechiness'] * 2.0 
    df['acousticness'] = df['acousticness'] * 1.5
    
    return df

with st.spinner('Chargement du catalogue...'):
    df = load_data()

if df.empty: st.stop()

# Fonctions pour générer les graphiques (Radar + 3D)
def make_radar_chart(target, recommended):
    categories = ['danceability', 'energy', 'acousticness', 'valence', 'speechiness']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[target[c] for c in categories], theta=categories, fill='toself', name='Ta sélection', line_color='#1DB954'))
    fig.add_trace(go.Scatterpolar(r=[recommended[c] for c in categories], theta=categories, fill='toself', name='Recommandation', line_color='#1E90FF'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=250, margin=dict(l=20, r=20, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), legend=dict(orientation="h", y=-0.2))
    return fig

def create_3d_plot(subset, target_vec, indices):
    background = subset.sample(min(200, len(subset)))
    fig = go.Figure()
    # Nuage de points global
    fig.add_trace(go.Scatter3d(x=background['danceability'], y=background['energy'], z=background['valence'], mode='markers', marker=dict(size=3, color='rgba(255, 255, 255, 0.1)'), name='Univers du genre', hoverinfo='none'))
    
    # Le point cible (YOU)
    target_idx = indices[0][0]
    t_row = subset.iloc[target_idx]
    fig.add_trace(go.Scatter3d(x=[t_row['danceability']], y=[t_row['energy']], z=[t_row['valence']], mode='markers+text', marker=dict(size=12, color='#1DB954'), name='Ta sélection', text=["YOU"], textposition="top center"))

    # Les recommandations
    rec_indices = indices[0][1:]
    rec_rows = subset.iloc[rec_indices]
    fig.add_trace(go.Scatter3d(x=rec_rows['danceability'], y=rec_rows['energy'], z=rec_rows['valence'], mode='markers', marker=dict(size=8, color='#1E90FF'), name='Recommandations', hovertext=rec_rows['track_name']))

    fig.update_layout(scene=dict(xaxis_title='Dansant', yaxis_title='Énergie', zaxis_title='Positivité', bgcolor='#0E1117'), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)', height=400)
    return fig

# Barre latérale pour la recherche et les infos
st.sidebar.title("🎧 Spotify Master")

st.sidebar.info(
    """
    **📊 À propos des données :**
    Ce projet utilise le dataset **Spotify Tracks** (Kaggle).
    
    - **114 000** Chansons
    - **114** Genres
    - Données extraites via l'API Spotify Developer.
    """
)
st.sidebar.markdown("---")

search_term = st.sidebar.text_input("🔍 Rechercher un chanteur", placeholder="ex: Michael Jackson")
filtered_df = df[df['search_label'].str.contains(search_term, case=False, na=False)].head(50) if search_term else pd.DataFrame()
selected_label = st.sidebar.selectbox("Sélectionne le son", options=filtered_df['search_label'].unique()) if not filtered_df.empty else None

st.sidebar.markdown("---")

# Section principale
# Petite explication pour l'utilisateur
with st.expander("ℹ️ Comment fonctionne cet algorithme ? (Cliquer pour comprendre)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🤖 L'Algorithme (KNN)")
        st.write("""
        Nous utilisons l'algorithme des **K-Plus Proches Voisins (KNN)**. 
        Imaginez chaque chanson comme un point dans un espace en 3 dimensions.
        L'algorithme calcule la **Distance Euclidienne** entre ta chanson et toutes les autres pour trouver celles qui sont mathématiquement les plus proches.
        """)
    with c2:
        st.markdown("### 📈 Le Vocabulaire Spotify")
        st.write("""
        - **Valence :** La "positivité" musicale (Triste 0.0 -> Joyeux 1.0).
        - **Energy :** Intensité et activité perçue.
        - **Danceability :** Facilité à danser sur le rythme.
        - **Acousticness :** Si le morceau est acoustique (vs électronique).
        - **Speechiness :** Présence de mots parlés (Rap/Podcast).
        """)

if selected_label:
    target_song = df[df['search_label'] == selected_label].iloc[0]
    
    st.markdown(f"## 🎶 {target_song['track_name']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Artiste", target_song['artists'])
    col2.metric("Genre", target_song['track_genre'])
    col3.metric("BPM", round(target_song['tempo']))
    col4.markdown(f"<br>", unsafe_allow_html=True)
    col4.link_button("▶️ Écouter sur Spotify", f"http://open.spotify.com/track/{target_song['track_id']}")
    
    if target_song['explicit']: st.caption("⚠️ Ce titre contient des paroles explicites.")
    st.markdown("---")

    # LOGIQUE KNN
    target_genre = target_song['track_genre']
    subset = df[df['track_genre'] == target_genre].copy()
    if target_song['explicit']: subset = subset[subset['explicit'] == True]
    if len(subset) < 10: subset = df[df['track_genre'] == target_genre].copy()
    subset = subset.reset_index(drop=True)

    feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = StandardScaler()
    X_subset = scaler.fit_transform(subset[feature_cols])
    
    target_in_subset = subset[subset['track_id'] == target_song['track_id']]
    
    if not target_in_subset.empty:
        target_idx = target_in_subset.index[0]
        base_vector = X_subset[target_idx].copy()
        target_vec = base_vector.reshape(1, -1)
        
        knn = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
        knn.fit(X_subset)
        distances, indices = knn.kneighbors(target_vec)
        
        tab_recos, tab_3d = st.tabs(["🔥 Recommandations", "🌌 Visualisation 3D"])
        
        with tab_recos:
            cols = st.columns(3)
            for i in range(1, len(indices[0])):
                idx = indices[0][i]
                rec_song = subset.iloc[idx]
                with cols[(i-1)%3]:
                    with st.container(border=True):
                        st.markdown(f"**{rec_song['track_name']}**")
                        st.text(f"{rec_song['artists']}")
                        delta_bpm = round(rec_song['tempo'] - target_song['tempo'])
                        st.metric("BPM", round(rec_song['tempo']), delta=delta_bpm)
                        st.markdown(f"[![Spotify](https://img.shields.io/badge/Spotify-Play-1DB954?style=flat&logo=spotify&logoColor=white)](http://open.spotify.com/track/{rec_song['track_id']})", unsafe_allow_html=True)
                        with st.expander("Voir l'ADN audio"):
                            radar = make_radar_chart(target_song, rec_song)
                            st.plotly_chart(radar, use_container_width=True, key=f"radar_{i}")

        with tab_3d:
            st.info("💡 Fais tourner le cube avec ta souris pour voir les positions !")
            chart_3d = create_3d_plot(subset, target_vec, indices)
            st.plotly_chart(chart_3d, use_container_width=True, key="chart_3d")
    else:
        st.error("Erreur technique.")
else:
    st.info("👈 Utilise la barre latérale pour chercher un son !")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #808080; padding-top: 10px; padding-bottom: 20px;'>
        <p>Conçu & Développé par <b>Alexandre Fiette</b></p>
        <p style='font-size: 0.8em;'>© 2025 - Tous droits réservés | Données issues de Kaggle & Spotify API</p>
    </div>
    """,
    unsafe_allow_html=True
)


