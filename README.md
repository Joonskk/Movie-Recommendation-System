# 🎬 Large-Scale Movie Recommendation System

A robust, memory-efficient movie recommendation pipeline built with Python, leveraging **Collaborative Filtering**, **Sparse Matrix Architecture (`SciPy csr_array`)**, and **Bayesian Average Statistical Smoothing** to process over 3.2M+ user ratings.

---

## 🚀 Key Features

* **Large-Scale Data Processing:** Engineered data pipelines to efficiently parse and analyze over **32 million ratings** across 200,000+ users and 84,000+ movies using Pandas and NumPy.
* **Memory-Optimized Sparse Matrix Architecture:** Implemented custom ID mapping dictionaries (`user_mapper`, `movie_mapper`) and converted utility matrices into **SciPy Compressed Sparse Row (`csr_array`)** format, achieving an extreme matrix sparsity of **0.19%** to minimize memory overhead.
* **Advanced Statistical Ranking (Bayesian Average):** Resolved ranking distortions caused by low-frequency ratings by implementing a Bayesian Average smoothing formula, accurately surfacing critically acclaimed titles over items with single skewed ratings.
* **Vectorized Data Aggregation:** Utilized NumPy vectorized operations (`np.diff` on index pointers) for high-performance retrieval and traversal of user-item interactions.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Manipulation & Analysis:** `pandas`, `NumPy`, `SciPy` (`csr_array`)
* **Data Visualization:** `Matplotlib`, `Seaborn`
* **Environment:** JupyterLab

---

## 📊 Methodology & Implementation

### 1. Exploratory Data Analysis (EDA)
* Analyzed global rating distributions, average user engagement, and movie frequency.
* Discovered raw rating anomalies (e.g., films with a 5.0 average driven by only a single review).

### 2. Bayesian Average Implementation
Applied statistical smoothing using the formula:
$$\text{Bayesian Average} = \frac{C \cdot m + \sum(x)}{C + n}$$
*(where C is the average dataset size per movie, m is the global average rating, and n is the review count).*

### 3. Utility Matrix & Collaborative Filtering Prep
* Constructed a high-dimensional sparse user-item matrix ($200,948 \times 84,432$) to set the foundation for neighborhood or latent-factor collaborative filtering.

---

## 💻 Getting Started

### Prerequisites
Make sure you have Python and JupyterLab installed along with the required libraries:
```bash
pip install pandas scikit-learn jupyter matplotlib seaborn fastapi uvicorn
