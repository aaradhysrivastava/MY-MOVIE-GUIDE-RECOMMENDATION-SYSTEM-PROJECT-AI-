import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path='data/ratings.csv'):
    return pd.read_csv(path)

def compute_similarity(ratings):
    user_ratings = ratings.pivot_table(index='user_name', columns='movie_title', values='rating').fillna(0)
    similarity = cosine_similarity(user_ratings)
    return pd.DataFrame(similarity, index=user_ratings.index, columns=user_ratings.index), user_ratings

def recommend(user_name, user_ratings, similarity, n_recommendations=5):
    sim_scores = similarity[user_name]
    weighted_ratings = user_ratings.T.dot(sim_scores) / sim_scores.sum()
    user_unseen = user_ratings.loc[user_name] == 0
    recommendations = weighted_ratings[user_unseen].sort_values(ascending=False).head(n_recommendations)
    return recommendations

def plot_recommendations(recommendations, user_name):
    plt.figure(figsize=(8, 5))
    recommendations.plot(kind='bar', color='skyblue')
    plt.title(f"Top {len(recommendations)} Movie Recommendations for {user_name}")
    plt.ylabel("Recommendation Score")
    plt.xlabel("Movie Title")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ratings = load_data()
    similarity, user_ratings = compute_similarity(ratings)
    user = 'Alice'  # Replace with a valid user from your dataset
    recommendations = recommend(user, user_ratings, similarity)
    
    print(f"Recommended movies for {user}:")
    for movie in recommendations.index:
        print("-", movie)
    
    # Show recommendations in a graph
    plot_recommendations(recommendations, user)
