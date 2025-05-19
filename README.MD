# 🎬 Movie Recommendation System

This project is a simple content-based and collaborative filtering recommendation system built using **Python** and **pandas**. It uses real user names and popular movie titles to generate personalized movie recommendations.

---

## 📌 Features

- 🔍 Recommends movies based on user preferences.
- 🤝 Uses collaborative filtering with cosine similarity.
- 📁 Works with a simple CSV dataset (`ratings.csv`).
- 🔧 Easy to modify and extend with real-world data.

---

## 🗂️ Dataset

The dataset is stored in `data/ratings.csv` and contains:

- `user_name` – Name of the user.
- `movie_title` – Title of the movie.
- `rating` – Rating given by the user (1 to 5).

**Example:**

```csv
user_name,movie_title,rating
Alice,The Shawshank Redemption,5
Bob,The Godfather,4
Carol,Pulp Fiction,4
```

---

## ⚙️ How It Works

1. 📥 Loads the dataset from `ratings.csv`.
2. 📊 Builds a user-item matrix.
3. 📐 Computes similarity between users using cosine similarity.
4. 🎯 Recommends movies a user hasn’t seen but similar users rated highly.

---

## 🧪 Installation & Usage

### 🔧 Requirements

Install required libraries:

```bash
pip install pandas scikit-learn
```

### ▶️ Run the Project

```bash
python main.py
```

You can change the **target user** inside `main.py` to get recommendations for others.

---

## 🔁 Example Output

```
Recommended movies for Alice:
- Pulp Fiction
- Forrest Gump
```

---

## 🛠️ Project Structure

```
recommendation-system/
├── data/
│   └── ratings.csv
├── main.py
├── dataset.py
└── README.md
```

---

## 🧠 Concepts Used

- Collaborative Filtering
- Cosine Similarity
- Pandas Pivot Tables
- Content-Based Recommendation (optional extension)


## 👨‍💻 Author

Made using Python by **[AARADHY SRIVASTAVA]**.
