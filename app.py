from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
similarity = cosine_similarity(tfidf_matrix)

def recommend(movie_title):
    idx = movies[movies['title'].str.contains(movie_title, case=False, na=False)].index
    if len(idx) == 0:
        return ["Movie not found"]
    idx = idx[0]
    scores = list(enumerate(similarity[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    recs = [movies.iloc[i[0]].title for i in sorted_scores]
    return recs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def get_recommendations():
    movie = request.args.get('movie')
    recs = recommend(movie)
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True)