# 🎬 Movie Recommendation System

A **content-based movie recommendation system** that suggests similar films using **TF-IDF Vectorization** and **Cosine Similarity** — built entirely without user ratings, relying instead on the intrinsic content of each film.

---

## 📌 Project Overview

Most recommendation engines you encounter (Netflix, Spotify, Amazon) use collaborative filtering — "users like you also watched...". This project takes a different, often more robust approach: **content-based filtering**. It analyses *what a movie is actually about* — its genres, cast, director, keywords, and tagline — and finds other movies with the most similar content profile.

| Item | Detail |
|------|--------|
| **Technique** | Content-Based Filtering |
| **Vectorisation** | TF-IDF (Term Frequency–Inverse Document Frequency) |
| **Similarity Metric** | Cosine Similarity |
| **Fuzzy Matching** | `difflib.get_close_matches` — handles typos gracefully |
| **Dataset** | [TMDB 5000 Movie Dataset – Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| **Output** | Top-N movie recommendations with similarity scores |

---

## 📂 Project Structure

```
movie_recommendation_system/
│
├── movie_recommendation_system.ipynb   # Jupyter Notebook (full walkthrough)
├── movie_recommendation_system.py      # Clean Python script
├── requirements.txt                    # Dependencies
├── movies.csv                          # Dataset (download from Kaggle)
├── eda_plots.png                       # Vote average, popularity, top directors
├── recommendations.png                 # Similarity score bar chart
└── README.md
```

---

## 🧠 How It Works

### Step 1 — Feature Selection
Five text-rich columns are chosen as content descriptors:

```
genres  +  keywords  +  tagline  +  cast  +  director
```

These are concatenated into a single string per movie. Missing values are replaced with `''` — they contribute nothing to the vector.

### Step 2 — TF-IDF Vectorisation

```
Raw text strings  ──▶  TF-IDF matrix  (shape: n_movies × n_unique_terms)
```

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into numerical vectors:
- **TF** — how often a word appears in *this* movie's description
- **IDF** — down-weights words that appear across *many* movies (common words carry less meaning)
- Result: distinctive words like a director's name or a niche genre term get higher weights

### Step 3 — Cosine Similarity

```
TF-IDF matrix  ──▶  Cosine Similarity matrix  (shape: n_movies × n_movies)
```

Cosine similarity measures the **angle** between two movie vectors:
- `1.0` → identical content profile
- `0.0` → completely dissimilar content

```
          A · B
cos(θ) = ———————
          |A||B|
```

### Step 4 — Fuzzy Title Matching

User input goes through `difflib.get_close_matches` before lookup, so a search for `"Dark Knight"` or even `"Dark Knght"` will still find `"The Dark Knight"`.

### Step 5 — Ranked Recommendations

Similarity scores for the query movie are sorted in descending order. The top-N results (excluding the movie itself) are returned as recommendations.

---

## 📊 Dataset Features Used

| Feature | Why It Matters |
|---------|---------------|
| `genres` | Broadest content signal — action, drama, sci-fi |
| `keywords` | Specific themes — time travel, heist, dystopia |
| `tagline` | Captures tone and marketing language |
| `cast` | Actor-based similarity — fans of an actor find related films |
| `director` | Auteur signal — directors have signature styles |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `movies.csv` from [Kaggle TMDB dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the project root.

### 4. Run
```bash
python movie_recommendation_system.py
```

The script will automatically recommend for `Avatar`, `The Dark Knight`, and `Inception`, then prompt you interactively.

---

## 🔄 Pipeline

```
movies.csv
    │
    ▼
EDA — vote average, popularity, top directors, missing values
    │
    ▼
Select 5 features: genres, keywords, tagline, cast, director
    │
    ▼
Fill NaN with '' → Concatenate into one string per movie
    │
    ▼
TF-IDF Vectorisation  (text → numerical matrix)
    │
    ▼
Cosine Similarity  (n_movies × n_movies similarity matrix)
    │
    ▼
User inputs a movie title
    │
    ▼
Fuzzy match (difflib) → find closest title
    │
    ▼
Sort similarity scores → return Top-N recommendations
    │
    ▼
Bar chart: similarity scores for recommendations
```

---

## 🍿 Example Output

```
Enter your favourite movie name: Inception

🎬 Closest match found: 'Inception'

🍿 Top 10 movies similar to 'Inception':

   1. The Dark Knight                              (similarity: 0.3821)
   2. Interstellar                                 (similarity: 0.3654)
   3. The Prestige                                 (similarity: 0.3412)
   4. Memento                                      (similarity: 0.3287)
   5. Batman Begins                                (similarity: 0.3101)
   ...
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — `TfidfVectorizer`, `cosine_similarity`
- **difflib** — fuzzy string matching for user input
- **seaborn / matplotlib** — EDA and recommendation visualization

---

## 🆚 Content-Based vs Collaborative Filtering

| Aspect | Content-Based (this project) | Collaborative Filtering |
|--------|------------------------------|------------------------|
| **Data needed** | Movie metadata only | User–item interaction history |
| **Cold start** | ✅ Works for new movies | ❌ Needs prior ratings |
| **Personalisation** | Based on movie similarity | Based on similar users |
| **Transparency** | ✅ Explainable (similar cast, genre) | Often a "black box" |
| **Serendipity** | Lower (stays within content niche) | Higher (cross-genre surprises) |

---

## 🚀 Future Improvements

- [ ] Add a **collaborative filtering** layer using user ratings for a hybrid system
- [ ] Weight features differently (e.g. director contributes more than tagline)
- [ ] Use **Word2Vec** or **BERT embeddings** instead of TF-IDF for richer semantic similarity
- [ ] Add popularity or recency as a tie-breaker for equally similar movies
- [ ] Build an interactive **Streamlit web app** with a search bar and movie poster display

---

## 📄 License

MIT License


