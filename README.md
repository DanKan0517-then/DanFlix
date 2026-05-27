# Danflix - Hybrid Movie Recommendation System
Danflix is an intelligent **Hybrid Movie Recommendation System** that combines multiple recommendation techniques to generate highly personalized movie suggestions.

Unlike traditional recommenders that rely on a single algorithm, Danflix integrates:

- **Collaborative Filtering (SVD)**
- **Deep Learning (Autoencoder)**
- **Association Rule Mining (Apriori)**


  ##  Recommendation Architecture

Danflix uses a **hybrid recommendation approach**.

### 1. Collaborative Filtering (SVD)
Uses **Singular Value Decomposition (SVD)** to predict missing user ratings based on collaborative behavior patterns.

**Purpose:**  
Learns hidden user preferences and movie similarities.

### 2. Deep Learning Autoencoder
A **Deep Autoencoder Neural Network** learns latent user-item interactions and reconstructs user preferences.

**Purpose:**  
Captures complex nonlinear relationships between users and movies.

### 3. Association Rule Mining (Apriori)
Uses **Apriori Algorithm** to identify movies that are commonly watched together.

Example:

```text
Users who watched:
Interstellar + Inception

Often also watched:
The Prestige
```
