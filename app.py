import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Leer imagen base64
encoded_image = Path("encoded_image.txt").read_text()

# Cargar datos
df = pd.read_csv("dogs_with_clusters_eng.csv")
df = df.dropna(subset=['descripcion', 'url']).copy()
df['dog_id'] = df.index

# Codificar variables categóricas
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(df[['tamaño', 'nivel_actividad', 'edad']])

# Vectorizar descripciones
vectorizer = TfidfVectorizer(max_features=100)
X_text = vectorizer.fit_transform(df['descripcion'].fillna(""))

# Combinar características
X = np.hstack((X_cat, X_text.toarray()))

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, df['dog_id'])

# Configurar Streamlit
st.set_page_config(page_title="PetMatch | Dog Adoption Recommender", page_icon="🐶", layout="wide")

# Estilos personalizados
st.markdown(f"""
    <style>
        html, body, .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stApp {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 1rem;
        }}
        h1 {{
            font-size: 3em;
            font-weight: bold;
            color: #222;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        h2, h3, label, .stTextInput > label, .stSelectbox > label, .stTextArea > label {{
            color: #222;
            font-weight: bold;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }}
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: #e8ffe8 !important;
            color: #222 !important;
            border: 1px solid #b4ddb4 !important;
            border-radius: 8px !important;
        }}
        .stSelectbox div[data-baseweb="select"] {{
            background-color: #e8ffe8 !important;
        }}
        button[kind="primary"], .stButton > button {{
            background-color: #88cc44 !important;
            color: white !important;
            font-weight: bold;
            border: none !important;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            transition: all 0.3s ease-in-out;
        }}
        .stButton > button:hover {{
            background-color: #76b83d !important;
            transform: scale(1.05);
        }}
        .match-card {{
            background-color: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
            margin-bottom: 1rem;
            color: #222 !important;
            transition: transform 0.3s ease;
        }}
        .match-card:hover {{
            transform: translateY(-5px);
        }}
        .match-card strong, .match-card a, .match-card p {{
            color: #222 !important;
        }}
        a {{
            text-decoration: none;
            color: #3388cc;
        }}
    </style>
""", unsafe_allow_html=True)

# Título
st.title("🐶 PetMatch: Find Your Perfect Dog")
st.subheader("Tell us about yourself and we’ll find the dog that fits your life!")

# Formulario
with st.form("profile_form"):
    lifestyle = st.text_input("How would you describe your lifestyle? (e.g. Active, Calm, Outdoorsy, Homebody)")
    activity_level = st.selectbox("What’s your daily physical activity level?", ["Sedentary", "Low", "Moderate", "High", "Very High"])
    home_size = st.selectbox("What size is your home?", ["Small", "Medium", "Large"])
    home_type = st.selectbox("What type of home do you live in?", ["Apartment", "House with garden", "Farm", "Rural"])
    location = st.selectbox("Where do you live?", ["City", "Suburbs", "Town", "Countryside"])
    household = st.selectbox("Who do you live with?", ["Alone", "Partner", "Family", "Roommates"])
    time_at_home = st.selectbox("How much time do you spend at home daily?", ["Few hours", "Several hours", "Most of the day"])
    hobbies = st.text_area("What are your hobbies or favorite activities?")
    submitted = st.form_submit_button("Find my new best friend!")

# Recomendaciones
if submitted:
    profile_text = f"{lifestyle} {activity_level} {home_size} {home_type} {location} {household} {time_at_home} {hobbies}"
    user_text_vec = vectorizer.transform([profile_text])
    user_cat_vec = np.zeros((1, X_cat.shape[1]))
    user_input = np.hstack((user_cat_vec, user_text_vec.toarray()))

    similarities = cosine_similarity(user_input, X)
    sim_scores = similarities[0]
    top_indices = np.argpartition(-sim_scores, range(10))[:10]
    top_sample = np.random.choice(top_indices, size=3, replace=False)
    top_matches = df.iloc[top_sample].copy()

    st.markdown("### 🐾 Your Top 3 Matches:")
    col_match1, col_match2, col_match3 = st.columns(3)
    columns = [col_match1, col_match2, col_match3]

    for col, (_, match) in zip(columns, top_matches.iterrows()):
        with col:
            name = match['nombre'] if pd.notnull(match['nombre']) else "Unnamed"
            location = match['ubicación'] if pd.notnull(match['ubicación']) else "Unknown"
            size = match['tamaño'] if pd.notnull(match['tamaño']) else "Unknown"
            age = match['edad'] if pd.notnull(match['edad']) else "Unknown"
            level = match['nivel_actividad'] if pd.notnull(match['nivel_actividad']) else "Unknown"
            description = match['descripcion'] if pd.notnull(match['descripcion']) and len(str(match['descripcion'])) < 400 else "No description available."
            url = match['url'] if pd.notnull(match['url']) else "#"

            st.markdown(f"""
            <div class='match-card'>
                <strong>🐾 {name}<strong><br>
                <strong>📍 Location:</strong> {location}<br>
                <strong>🔏 Size:</strong> {size}<br>
                <strong>🎂 Age:</strong> {age}<br>
                <strong>🏃 Activity Level:</strong> {level}<br><br>
                <strong>📄 Description:</strong><br>{description}<br><br>
                <a href="{url}" target="_blank">🔗 View Full Profile</a>
            </div>
            """, unsafe_allow_html=True)
