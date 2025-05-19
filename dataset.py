import pandas as pd
import os

# Create data folder if not exists
os.makedirs('data', exist_ok=True)

# Sample user-movie ratings data with real names and movie titles
data = {
    'user_name': [
        'Alice', 'Alice', 'Alice',
        'Bob', 'Bob', 'Bob',
        'Carol', 'Carol', 'Carol'
    ],
    'movie_title': [
        'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
        'The Shawshank Redemption', 'Pulp Fiction', 'Forrest Gump',
        'The Godfather', 'Pulp Fiction', 'The Dark Knight'
    ],
    'rating': [5, 4, 5, 4, 5, 3, 5, 4, 4]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/ratings.csv', index=False)

print("ratings.csv dataset with real names and movies created successfully!")
