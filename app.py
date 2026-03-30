import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# Load and preprocess data
# -------------------------------
movies = pd.read_csv("movies.csv")

movies = movies.dropna(subset=['title'])
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['keywords'] = movies['keywords'].fillna('')

# Combine features
movies['tags'] = (
    movies['overview'] + " " +
    movies['genres'] + " " +
    movies['keywords']
)

movies['tags'] = movies['tags'].apply(lambda x: str(x).lower())

# Optional filter
if 'vote_average' in movies.columns:
    movies = movies[movies['vote_average'] > 6]

# -------------------------------
# Vectorization (optimized)
# -------------------------------
cv = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags'])

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie):
    movie = movie.lower()

    if movie not in movies['title'].str.lower().values:
        return "❌ Movie not found"

    index = movies[movies['title'].str.lower() == movie].index[0]

    distances = cosine_similarity(vectors[index], vectors).flatten()

    movies_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )[1:10]

    recommended = []

    for i in movies_list:
        title = movies.iloc[i[0]].title

        if "alien" in title.lower():
            continue

        recommended.append(title)

        if len(recommended) == 5:
            break

    return "\n".join(recommended)

# -------------------------------
# Gradio UI
# -------------------------------
app = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(label="Enter Movie Name"),
    outputs=gr.Textbox(label="Recommended Movies"),
    title="🎬 Movie Recommendation System",
    description="Enter a movie name and get similar movie recommendations!"
)

app.launch()
