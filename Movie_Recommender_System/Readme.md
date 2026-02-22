# 🎬 Movie Recommender System

A **Content-Based Movie Recommender System** built using Python that suggests movies similar to a given title based on a combination of plot overviews, genres, keywords, cast, and director information. The system uses **Natural Language Processing (NLP)** and **Cosine Similarity** to find and rank the most similar movies from a dataset of ~4,800 films sourced from TMDB (The Movie Database).

---

## 🔄 System Workflow Overview

![Content-Based Movie Recommender System Workflow]([images/workflow.png](https://raw.githubusercontent.com/Aayush005-netizen/Data-Science-Machine-Learning-Projects/main/Images/workflow.png))

---

## 📌 Types of Recommender Systems

Before diving into the project, it's important to understand the three common approaches to building recommender systems:

1. **Content-Based Filtering** — Recommends items similar to what a user has liked before, based on item attributes (e.g., genre, cast, keywords). *This project uses this approach.*
2. **Collaborative Filtering** — Identifies users with similar behavior and recommends what those users liked.
3. **Hybrid** — Combines both content-based and collaborative filtering for more robust recommendations.

---

## 🗂️ Project Workflow

### 1. Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

The foundational libraries are imported at the start. `pandas` and `numpy` handle data loading and manipulation, while `matplotlib` is available for any exploratory visualization. Loading these upfront keeps the notebook organized and makes dependencies explicit.

---

### 2. Datasets

#### 2a. Importing

Two CSV files from the TMDB 5000 Movie Dataset are loaded:

- **`tmdb_5000_movies.csv`** — Contains movie metadata such as genres, keywords, overview, budget, revenue, etc.
- **`tmdb_5000_credits.csv`** — Contains cast and crew information for each movie.

These two files are loaded separately because they originate from different data sources and must be combined before use.

#### 2b. Merging Datasets

```python
movies = movies.merge(credits, on='title')
```

The two datasets are merged on the `title` column. This joins the movie metadata with its corresponding cast and crew information into a single unified DataFrame, giving us all the features we need in one place.

---

### 3. Preprocessing

#### 3a. Removing Unnecessary Columns

After merging, the combined DataFrame has 23 columns — many of which are irrelevant to building a content-based recommender (e.g., `budget`, `revenue`, `runtime`, `homepage`). Only the 7 most meaningful columns are retained:

```
movie_id, title, overview, genres, keywords, cast, crew
```

Keeping only relevant columns reduces noise and makes subsequent processing faster and cleaner.

#### 3b. Handling Missing Values and Duplicates

- **Null check**: `overview` had 3 missing values. Since the overview is a critical part of the tags, rows with missing overviews are dropped using `dropna()`.
- **Duplicate check**: No duplicate rows were found.

This ensures the model trains on clean, complete data.

#### 3c. Making Tags

The core idea of this system is to build a single "tag" string per movie that encodes all its important characteristics. Each column requires special handling because their values are stored as stringified JSON lists.

**genres & keywords** — A helper function using Python's `ast.literal_eval` parses the stringified list of dictionaries and extracts just the `name` field from each entry.

**cast** — A similar helper is used, but with a limit of **3 actors** per movie. This is intentional — the top-billed cast members are most representative of the film, and including too many actors would dilute the signal.

**crew** — Only the **Director** is extracted from the crew list. The director has the most defining influence on a film's style and tone, making them the most valuable crew signal for recommendation.

**overview** — The plot summary is split into a list of individual words using `.split()` to match the format of the other tag components.

#### 3d. Removing Spaces Between Names

Multi-word names like "Sam Worthington" or "Science Fiction" are concatenated into single tokens (`SamWorthington`, `ScienceFiction`). This is a critical step — without it, the vectorizer would treat "Sam" and "Worthington" as separate, unrelated words, losing the identity of the person entirely.

#### 3e. Concatenating All Columns into Tags

```python
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
```

All processed lists are concatenated into a single `tags` list per movie. This unified tag is the "fingerprint" of the movie that the model will use for comparison.

A clean new DataFrame `new_df` is created with only `movie_id`, `title`, and `tags`.

#### 3f. Converting Tags from List to String

```python
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
```

The tags list is joined into a single space-separated string. This is necessary because the vectorizer (used in the next step) works on string inputs, not Python lists.

---

### 4. Vectorization

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
# Shape: (4806, 5000)
```

**CountVectorizer** converts the tag strings into numerical vectors. Each movie becomes a vector of word counts across the top 5,000 most frequent words in the corpus. Common English stop words (like "the", "a", "is") are excluded because they carry no meaningful information for finding similarity between movies.

The result is a **4806 × 5000** matrix, where each row is a movie and each column is a word frequency count.

---

### 5. Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
```

Cosine similarity measures the angle between two vectors — a score of `1.0` means the movies are identical in tag composition, and `0.0` means they share nothing in common. It is preferred over Euclidean distance here because it is **not affected by the magnitude** of the vectors (i.e., a long movie description vs. a short one won't unfairly skew results — only the direction/pattern of words matters).

The output is a **4806 × 4806** matrix where every cell `[i][j]` represents how similar movie `i` is to movie `j`.

---

### 6. Recommender System

```python
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
```

The `recommend()` function:
1. Finds the index of the queried movie in the DataFrame.
2. Retrieves its similarity scores against all other movies from the precomputed similarity matrix.
3. Sorts movies by similarity score in descending order.
4. Prints the **top 5 most similar movies** (skipping index `0`, which is the movie itself).

**Example Output for `recommend("Gandhi")`:**
```
Gandhi, My Father
The Wind That Shakes the Barley
A Passage to India
Guiana 1838
Bloody Sunday
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | CountVectorizer & Cosine Similarity |
| ast | Parsing stringified JSON columns |
| Google Colab | Development environment |

---

## 📁 Dataset

- **Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Files**: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
- **Size**: ~4,806 movies after cleaning

---

## 🚀 How to Run

1. Clone the repository and open `Movie_Recommender_System.ipynb` in Jupyter or Google Colab.
2. Download the TMDB 5000 dataset and update the file paths accordingly.
3. Run all cells in order.
4. Call the `recommend()` function with any valid movie title:

```python
recommend("The Dark Knight")
```
