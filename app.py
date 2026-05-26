"""
Flask Web Server for DanFlix Hybrid Recommender System.
"""
import os
from flask import Flask, request, jsonify, render_template
from recommender import HybridRecommender

app = Flask(__name__, template_folder="templates", static_folder="static")

# Lazy-load the recommender so the app starts even if models aren't trained yet
recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        models_dir = "models"
        # Check if model files exist
        required_files = ["movies.pkl", "movie2idx.pkl", "idx2movie.pkl", 
                          "svd_item_factors.npy", "movie_embeddings.npy"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(models_dir, f))]
        
        if missing:
            raise FileNotFoundError(f"Missing trained model files: {missing}. Please run train_models.py first.")
            
        recommender = HybridRecommender(models_dir=models_dir)
    return recommender

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip().lower()
    if not query:
        return jsonify([])
        
    try:
        rec = get_recommender()
        # Find matching movies (limit to 10 suggestions)
        matches = rec.movies[rec.movies['title'].str.lower().str.contains(query, na=False)].head(10)
        
        results = []
        for _, row in matches.iterrows():
            results.append({
                'movieId': int(row['movieId']),
                'title': str(row['title']),
                'genres': str(row['genres'])
            })
        return jsonify(results)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json() or {}
    
    movie_id = data.get("movieId")
    if movie_id is None:
        return jsonify({"error": "Missing parameter 'movieId'"}), 400
        
    try:
        movie_id = int(movie_id)
        w_svd = float(data.get("w_svd", 0.40))
        w_ae = float(data.get("w_ae", 0.35))
        w_apriori = float(data.get("w_apriori", 0.25))
        
        rec = get_recommender()
        recommendations = rec.recommend(movie_id, w_svd=w_svd, w_ae=w_ae, w_apriori=w_apriori, top_n=10)
        
        # Get target movie details
        target_title = rec.id2title.get(movie_id, "Unknown Movie")
        target_genres = rec.movies[rec.movies['movieId'] == movie_id]['genres'].values[0] if movie_id in rec.id2title else ""
        
        return jsonify({
            'target': {
                'movieId': movie_id,
                'title': target_title,
                'genres': target_genres
            },
            'recommendations': recommendations
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Run server on port 5000 in debug/reloading mode
    app.run(host="127.0.0.1", port=5000, debug=True)
