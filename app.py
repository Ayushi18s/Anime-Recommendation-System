import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')

# Initialize stemmer
ps = PorterStemmer()

# Stemmer function
def stem_text(text):
    words = nltk.word_tokenize(text.lower())
    stemmed_words = [ps.stem(word) for word in words]
    return " ".join(stemmed_words)

# Load and clean dataset
@st.cache_data
def load_data():
    anime = pd.read_csv("C:/Users/HP/Downloads/anime recommendation system/anime.csv")
    anime = anime.dropna(subset=["name", "genre"])
    anime = anime.drop_duplicates(subset="name")
    anime["stemmed_name"] = anime["name"].apply(stem_text)
    
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(anime["genre"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return anime, cosine_sim

anime, cosine_sim = load_data()

# Recommendation function
def get_recommendations(input_title):
    stemmed_input = stem_text(input_title)
    match = anime[anime["stemmed_name"] == stemmed_input]

    if match.empty:
        return []

    idx = match.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    anime_indices = [i[0] for i in sim_scores]
    return anime["name"].iloc[anime_indices].tolist()

# --- Streamlit UI ---
st.title("🎌 Anime Recommender")
st.write("Find similar anime based on genres.")

# Input
anime_input = st.text_input("Enter an anime title:")

if anime_input:
    recommendations = get_recommendations(anime_input)

    if recommendations:
        st.subheader("You might also like:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("Anime not found. Please try a different title.")

