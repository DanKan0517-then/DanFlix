"""
Hybrid Recommender Engine for DanFlix.
Combines SVD Collaborative Filtering, PyTorch Autoencoder taste mappings, and Apriori association rules.
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class HybridRecommender:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Load metadata and maps
        with open(os.path.join(models_dir, "movies.pkl"), "rb") as f:
            self.movies = pickle.load(f)
        with open(os.path.join(models_dir, "movie2idx.pkl"), "rb") as f:
            self.movie2idx = pickle.load(f)
        with open(os.path.join(models_dir, "idx2movie.pkl"), "rb") as f:
            self.idx2movie = pickle.load(f)
            
        # Movie ID to title and vice versa
        self.id2title = dict(zip(self.movies['movieId'], self.movies['title']))
        self.title2id = dict(zip(self.movies['title'], self.movies['movieId']))
        
        # Load SVD embeddings
        self.svd_factors = np.load(os.path.join(models_dir, "svd_item_factors.npy"))
        # Normalize SVD factors for cosine similarity
        norms = np.linalg.norm(self.svd_factors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.svd_factors_norm = self.svd_factors / norms
        
        # Load Autoencoder embeddings
        self.ae_embeddings = np.load(os.path.join(models_dir, "movie_embeddings.npy"))
        # Normalize Autoencoder embeddings for cosine similarity
        norms_ae = np.linalg.norm(self.ae_embeddings, axis=1, keepdims=True)
        norms_ae[norms_ae == 0] = 1.0
        self.ae_embeddings_norm = self.ae_embeddings / norms_ae
        
        # Load Apriori rules
        self.apriori_rules = None
        apriori_path = os.path.join(models_dir, "apriori_rules.pkl")
        if os.path.exists(apriori_path):
            with open(apriori_path, "rb") as f:
                self.apriori_rules = pickle.load(f)
        else:
            print("Warning: apriori_rules.pkl not found. Apriori scores will be 0.")

    def get_svd_similarities(self, movie_idx):
        # Cosine similarity is dot product of normalized embeddings
        target_embedding = self.svd_factors_norm[movie_idx].reshape(1, -1)
        sims = (self.svd_factors_norm @ target_embedding.T).squeeze()
        # Scale from [-1, 1] to [0, 1]
        sims = (sims + 1.0) / 2.0
        return sims

    def get_ae_similarities(self, movie_idx):
        target_embedding = self.ae_embeddings_norm[movie_idx].reshape(1, -1)
        sims = (self.ae_embeddings_norm @ target_embedding.T).squeeze()
        # Scale from [-1, 1] to [0, 1]
        sims = (sims + 1.0) / 2.0
        return sims

    def get_apriori_scores(self, movie_title):
        scores = np.zeros(len(self.movie2idx))
        if self.apriori_rules is None or len(self.apriori_rules) == 0:
            return scores
            
        # Find rules where antecedent is the target movie
        # Antecedents in mlxtend rules are frozensets
        matching_rules = self.apriori_rules[
            self.apriori_rules['antecedents'].apply(lambda x: movie_title in x)
        ]
        
        if len(matching_rules) == 0:
            return scores
            
        # Aggregate consequents and their lifts
        consequent_lifts = {}
        for _, row in matching_rules.iterrows():
            for consequent in row['consequents']:
                # Save the maximum lift for this consequent
                lift = row['lift']
                consequent_lifts[consequent] = max(consequent_lifts.get(consequent, 0.0), lift)
                
        if not consequent_lifts:
            return scores
            
        # Normalize lifts to [0, 1] range using min-max scaling of the matching set
        max_lift = max(consequent_lifts.values())
        min_lift = min(consequent_lifts.values())
        lift_range = max_lift - min_lift if max_lift > min_lift else 1.0
        
        for title, lift in consequent_lifts.items():
            if title in self.title2id:
                mid = self.title2id[title]
                if mid in self.movie2idx:
                    m_idx = self.movie2idx[mid]
                    # Normalized lift score
                    scores[m_idx] = (lift - min_lift) / lift_range
                    
        return scores

    def recommend(self, movie_id, w_svd=0.40, w_ae=0.35, w_apriori=0.25, top_n=10):
        if movie_id not in self.movie2idx:
            raise ValueError(f"Movie ID {movie_id} not in trained recommendation mappings.")
            
        target_idx = self.movie2idx[movie_id]
        target_title = self.id2title[movie_id]
        
        # 1. Normalize weights
        total_w = w_svd + w_ae + w_apriori
        if total_w == 0:
            w_svd, w_ae, w_apriori = 0.33, 0.33, 0.33
            total_w = 1.0
        w_svd /= total_w
        w_ae /= total_w
        w_apriori /= total_w
        
        # 2. Get scores from SVD
        svd_scores = self.get_svd_similarities(target_idx)
        
        # 3. Get scores from Autoencoder
        ae_scores = self.get_ae_similarities(target_idx)
        
        # 4. Get scores from Apriori
        apriori_scores = self.get_apriori_scores(target_title)
        
        # 5. Combine scores
        combined_scores = (w_svd * svd_scores) + (w_ae * ae_scores) + (w_apriori * apriori_scores)
        
        # Create recommendation dataframe
        rec_df = pd.DataFrame({
            'movieId': [self.idx2movie[i] for i in range(len(self.movie2idx))],
            'title': [self.id2title[self.idx2movie[i]] for i in range(len(self.movie2idx))],
            'genres': [self.movies.iloc[i]['genres'] for i in range(len(self.movie2idx))],
            'svd_score': svd_scores,
            'ae_score': ae_scores,
            'apriori_score': apriori_scores,
            'combined_score': combined_scores
        })
        
        # Remove the input movie itself
        rec_df = rec_df[rec_df['movieId'] != movie_id]
        
        # Get top recommendations sorted by combined score
        top_recs = rec_df.sort_values(by='combined_score', ascending=False).head(top_n)
        
        # Format return payload
        results = []
        for _, row in top_recs.iterrows():
            results.append({
                'movieId': int(row['movieId']),
                'title': str(row['title']),
                'genres': str(row['genres']),
                'svd_score': float(row['svd_score']),
                'ae_score': float(row['ae_score']),
                'apriori_score': float(row['apriori_score']),
                'combined_score': float(row['combined_score'])
            })
            
        return results
