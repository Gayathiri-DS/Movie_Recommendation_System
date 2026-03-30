import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")

    movies = movies.dropna(subset=['title'])
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')
    movies['keywords'] = movies['keywords'].fillna('')

    movies['tags'] = (
        movies['overview'] + " " +
        movies['genres'] + " " +
        movies['keywords']
    )

    movies['tags'] = movies['tags'].apply(lambda x: str(x).lower())

    return movies


@st.cache_resource
def create_vectors(movies):
    from sklearn.feature_extraction.text import TfidfVectorizer

    cv = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'])

    return vectors


movies = load_data()
vectors = create_vectors(movies)
def recommend(movie):
    movie = movie.lower()

    if movie not in movies['title'].str.lower().values:
        return ["Movie not found"]

    index = movies[movies['title'].str.lower() == movie].index[0]

    distances = cosine_similarity(vectors[index], vectors).flatten()

    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:20]

    recommended = []

    for i in movies_list:
        title = movies.iloc[i[0]].title

        if "alien" in title.lower():
            continue

        recommended.append(title)

        if len(recommended) == 5:
            break

    return recommended
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
