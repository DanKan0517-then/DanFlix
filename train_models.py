"""
Train SVD and PyTorch Autoencoder Models on MovieLens 20M dataset.
"""
import os
import time
import pickle
import warnings
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MOVIE_CSV = "movie.csv"
RATING_CSV = "rating.csv"
ZIP_PATH = "ml-20m.zip"

def download_dataset():
    if not os.path.exists(MOVIE_CSV) or not os.path.exists(RATING_CSV):
        # Look in subfolder first
        if os.path.exists("ml-20m/movies.csv") and os.path.exists("ml-20m/ratings.csv"):
            print("Found dataset files in ml-20m subfolder. Copying...")
            import shutil
            shutil.copy("ml-20m/movies.csv", MOVIE_CSV)
            shutil.copy("ml-20m/ratings.csv", RATING_CSV)
            return

        if not os.path.exists(ZIP_PATH):
            print("MovieLens dataset not found. Downloading ml-20m.zip (~325MB)...")
            url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
            urllib.request.urlretrieve(url, ZIP_PATH)
            print("Download completed.")
        
        print("Extracting ml-20m.zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        import shutil
        shutil.copy("ml-20m/movies.csv", MOVIE_CSV)
        shutil.copy("ml-20m/ratings.csv", RATING_CSV)
        print("Dataset setup completed successfully.")

# Initialize dataset
download_dataset()

print("Loading dataset...")
movies_df = pd.read_csv(MOVIE_CSV)
ratings_df = pd.read_csv(RATING_CSV)

print(f"Original sizes: {len(movies_df):,} movies, {len(ratings_df):,} ratings.")

# Deterministic 10% sample
print("Sampling 10% of ratings for performance...")
ratings = ratings_df.sample(frac=0.10, random_state=42).reset_index(drop=True)

# Keep only movies that are in the ratings sample
unique_movies = ratings['movieId'].unique()
movies = movies_df[movies_df['movieId'].isin(unique_movies)].reset_index(drop=True)
print(f"Sample contains: {ratings['userId'].nunique():,} users, {len(movies):,} movies, {len(ratings):,} ratings.")

# Create maps
user_ids = sorted(ratings['userId'].unique())
movie_ids = sorted(movies['movieId'].unique())

user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
movie2idx = {mid: idx for idx, mid in enumerate(movie_ids)}
idx2movie = {idx: mid for mid, idx in movie2idx.items()}

# Save maps and movies metadata
with open(os.path.join(MODELS_DIR, "user2idx.pkl"), "wb") as f:
    pickle.dump(user2idx, f)
with open(os.path.join(MODELS_DIR, "movie2idx.pkl"), "wb") as f:
    pickle.dump(movie2idx, f)
with open(os.path.join(MODELS_DIR, "idx2movie.pkl"), "wb") as f:
    pickle.dump(idx2movie, f)
with open(os.path.join(MODELS_DIR, "movies.pkl"), "wb") as f:
    pickle.dump(movies, f)

# Map ids to indices in ratings
ratings['u_idx'] = ratings['userId'].map(user2idx)
ratings['m_idx'] = ratings['movieId'].map(movie2idx)

n_users = len(user_ids)
n_movies = len(movie_ids)

print(f"Matrix shape: {n_users} users x {n_movies} movies")

# 2. Train SVD
print("\n--- Training SVD ---")
# Construct sparse user-item matrix
row = ratings['u_idx'].values
col = ratings['m_idx'].values
data = ratings['rating'].values

# Calculate user means for centering
user_sums = np.zeros(n_users)
user_counts = np.zeros(n_users)
for u, r in zip(row, data):
    user_sums[u] += r
    user_counts[u] += 1
user_means = np.zeros(n_users)
non_zero_users = user_counts > 0
user_means[non_zero_users] = user_sums[non_zero_users] / user_counts[non_zero_users]

# Centered data
centered_data = data - user_means[row]
sparse_rating_matrix = csr_matrix((centered_data, (row, col)), shape=(n_users, n_movies), dtype=np.float32)

print("Computing Truncated SVD (k=100)...")
U, sigma, Vt = svds(sparse_rating_matrix, k=100)

# Sort singular values in descending order
idx = np.argsort(sigma)[::-1]
U = U[:, idx]
sigma = sigma[idx]
Vt = Vt[idx, :]

# Calculate item factors (movie embeddings)
# We want Vt.T * sigma
svd_item_factors = Vt.T * sigma

# Save SVD factors
np.save(os.path.join(MODELS_DIR, "svd_item_factors.npy"), svd_item_factors)

# Save SVD model structures
svd_model = {
    'U': U,
    'sigma': sigma,
    'Vt': Vt,
    'user_means': user_means
}
with open(os.path.join(MODELS_DIR, "svd_model.pkl"), "wb") as f:
    pickle.dump(svd_model, f)
print("SVD training completed and saved.")

# 3. Train Autoencoder
print("\n--- Training Autoencoder ---")
# PyTorch device check (use MPS on Apple Silicon, CUDA on GPU server, else CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Build dense ratings matrix for training (normalized to [0,1])
# We will use csr_matrix to build it efficiently, then load in batches
normalized_ratings = data / 5.0
train_matrix = csr_matrix((normalized_ratings, (row, col)), shape=(n_users, n_movies), dtype=np.float32)

# Define PyTorch Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

# Initialize model
latent_dim = 128
model = Autoencoder(n_movies, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

# Masked MSE Loss function (only evaluate ratings that are non-zero)
def masked_mse_loss(reconstructed, target):
    mask = (target > 0).float()
    loss = nn.functional.mse_loss(reconstructed * mask, target * mask, reduction='sum')
    # Normalise by number of non-zero elements
    num_ratings = torch.clamp(mask.sum(), min=1.0)
    return loss / num_ratings

# Custom data loader for sparse matrix
class SparseDataset(torch.utils.data.Dataset):
    def __init__(self, sparse_matrix):
        self.matrix = sparse_matrix

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        # Convert row to dense float32 array
        row_dense = self.matrix[idx].toarray().squeeze()
        return torch.tensor(row_dense, dtype=torch.float32)

dataset = SparseDataset(train_matrix)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Training loop
epochs = 5  # Quick training that still converges sufficiently
print(f"Training for {epochs} epochs...")
model.train()
for epoch in range(epochs):
    start_time = time.time()
    epoch_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        _, reconstructed = model(batch)
        loss = masked_mse_loss(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch.size(0)
    
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f} | Time: {time.time() - start_time:.2f}s")

# Save model weights
torch.save(model.state_dict(), os.path.join(MODELS_DIR, "autoencoder.pt"))
print("Autoencoder weights saved.")

# Generate latent vectors for all users and compute movie embeddings
print("Generating movie embeddings from Autoencoder...")
model.eval()
all_latents = []

# Process users in batches
eval_dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
with torch.no_grad():
    for batch in eval_dataloader:
        batch = batch.to(device)
        latents, _ = model(batch)
        all_latents.append(latents.cpu().numpy())

all_latents = np.concatenate(all_latents, axis=0) # Shape: (n_users, 128)

# Compute movie embeddings:
# For each movie j, embedding is average user latent weighted by rating of user for movie j.
# user_item_norm is (n_users, n_movies).
# movie_embeddings = user_item_norm.T @ all_latents.
# To avoid memory errors, do this multiplication in chunks.
print("Aggregating user latents into movie embeddings...")
movie_embeddings = np.zeros((n_movies, latent_dim), dtype=np.float32)

# Normalise the columns of the train_matrix (user ratings for each movie)
# We want the weights to sum to 1 for each movie.
col_sums = np.array(train_matrix.sum(axis=0)).squeeze()
col_sums[col_sums == 0] = 1.0  # Avoid division by zero

# Compute weighted average
chunk_size = 1000
for i in range(0, n_movies, chunk_size):
    end_idx = min(i + chunk_size, n_movies)
    # Extract submatrix for chunk of movies: shape (n_users, chunk_len)
    sub_matrix = train_matrix[:, i:end_idx].toarray()
    # Compute dot product: shape (chunk_len, 128)
    chunk_embeddings = sub_matrix.T @ all_latents
    # Divide by col sums
    chunk_embeddings /= col_sums[i:end_idx, np.newaxis]
    movie_embeddings[i:end_idx] = chunk_embeddings

# Save Autoencoder movie embeddings
np.save(os.path.join(MODELS_DIR, "movie_embeddings.npy"), movie_embeddings)
print("Autoencoder movie embeddings saved successfully.")
print("\nModel training pipeline complete!")
