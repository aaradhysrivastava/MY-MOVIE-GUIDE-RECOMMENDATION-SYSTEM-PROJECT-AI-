import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

ratings = pd.read_csv('data/ratings.csv')
print("Columns in ratings.csv:", ratings.columns.tolist())
print(ratings.head())

def load_data(path='data/ratings.csv'):
    return pd.read_csv(path)

def compute_similarity(ratings):
    # Create user-movie rating matrix (users as rows, movies as columns)
    user_ratings = ratings.pivot_table(index='user_name', columns='movie_title', values='rating').fillna(0)
    similarity = cosine_similarity(user_ratings)
    return pd.DataFrame(similarity, index=user_ratings.index, columns=user_ratings.index)

def recommend(user_name, ratings, similarity, n_recommendations=5):
    user_ratings = ratings.pivot_table(index='user_name', columns='movie_title', values='rating').fillna(0)
    sim_scores = similarity[user_name]
    weighted_ratings = user_ratings.T.dot(sim_scores) / sim_scores.sum()
    # Recommend movies user hasn't rated yet
    user_unseen = user_ratings.loc[user_name] == 0
    recommendations = weighted_ratings[user_unseen].sort_values(ascending=False).head(n_recommendations)
    return recommendations.index.tolist()

if __name__ == '__main__':
    ratings = load_data()
    similarity = compute_similarity(ratings)
    user = 'Alice'  # Change to any user name from your data
    recs = recommend(user, ratings, similarity)
    print(f"Recommended movies for {user}:")
    for movie in recs:
        print("-", movie)
