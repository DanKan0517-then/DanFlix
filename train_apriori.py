"""
Quick script to train just the Apriori model (SVD + AE already saved).
"""
import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
RATING_THRESHOLD = 3.5
TOP_N_MOVIES = 200
MIN_LIKES_PER_USER = 5

print("--- Apriori Association Rule Mining ---")

# Load mapping and movies metadata from previous script
with open(os.path.join(MODELS_DIR, "movies.pkl"), "rb") as f:
    movies = pickle.load(f)
with open(os.path.join(MODELS_DIR, "movie2idx.pkl"), "rb") as f:
    movie2idx = pickle.load(f)

# Load rating data
print("Loading ratings for Apriori...")
rating_df = pd.read_csv("rating.csv")

# Filter ratings to include only movies in our sample
rating_df = rating_df[rating_df['movieId'].isin(movie2idx.keys())]

# 1. Identify the top N most popular movies
print(f"Finding top {TOP_N_MOVIES} most popular movies...")
popular_movies = rating_df.groupby('movieId').size().nlargest(TOP_N_MOVIES).index.tolist()

# 2. Filter ratings for only those popular movies and positive ratings (rating >= 3.5)
liked_ratings = rating_df[(rating_df['movieId'].isin(popular_movies)) & (rating_df['rating'] >= RATING_THRESHOLD)]

# 3. Keep only users who liked at least MIN_LIKES_PER_USER of these popular movies
user_counts = liked_ratings['userId'].value_counts()
dense_users = user_counts[user_counts >= MIN_LIKES_PER_USER].index.tolist()
liked_ratings = liked_ratings[liked_ratings['userId'].isin(dense_users)]

print(f"Mined transaction space: {len(dense_users):,} users, {len(popular_movies):,} popular movies, {len(liked_ratings):,} likes.")

# 4. Pivot into transactional baskets: users as index, movie indices as columns, 1 if liked, 0 otherwise
print("Creating binary transaction matrix...")
# Map movie IDs to their movie titles for readable rules
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
liked_ratings['movie_title'] = liked_ratings['movieId'].map(movie_id_to_title)

# Create a crosstab (efficient way to build a binary basket)
basket = pd.crosstab(liked_ratings['userId'], liked_ratings['movie_title'])
basket = (basket > 0).astype(bool)

# 5. Run Apriori
print("Running Apriori frequent itemset mining (min_support=0.01)...")
start_time = time.time()
freq_items = apriori(basket, min_support=0.01, use_colnames=True, max_len=2)
print(f"  Found {len(freq_items):,} frequent itemsets in {time.time() - start_time:.2f}s")

# 6. Extract association rules
rules = None
if len(freq_items) > 0:
    print("Generating association rules (min_confidence=0.1)...")
    try:
        # Pass num_itemsets parameter to prevent warning/error in newer versions of mlxtend
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.1, num_itemsets=len(freq_items))
    except TypeError:
        # Fallback if older mlxtend version does not accept num_itemsets
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
    
    print(f"  Mined {len(rules):,} association rules.")
    
    # Save the association rules
    with open(os.path.join(MODELS_DIR, "apriori_rules.pkl"), "wb") as f:
        pickle.dump(rules, f)
    print("Apriori association rules saved.")
else:
    print("Error: No frequent itemsets found. Lower min_support or increase TOP_N_MOVIES.")

print("Apriori training script finished.")
