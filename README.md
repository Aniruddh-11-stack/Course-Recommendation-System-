# Course Recommendation System

This repository contains a Course Recommendation System built using Python. It analyzes a Coursera dataset and provides course recommendations based on user interactions and content similarities. 

## Features

The project demonstrates three different approaches to building recommendation systems:

1. **Content-Based Filtering (Cosine Similarity)**: Recommends courses similar to ones a user has already expressed interest in, based on course tags (Name, Difficulty, Description, Skills).
2. **Item-to-Item Collaborative Filtering**: Uses quantitative metadata (Course Rating, Difficulty, University) to scale and find mathematically similar courses using Cosine Similarity.
3. **Deep Learning (Autoencoder)**: Uses a TensorFlow/Keras Unsupervised Autoencoder to learn efficient 2D embeddings of course features, capturing non-linear similarities.

## Dataset

The system uses `Coursera.csv`, which contains details such as:
- `Course Name`
- `University`
- `Difficulty Level`
- `Course Rating`
- `Course Description`
- `Skills`

*(Note: The `Coursera.csv` dataset must be present in the root directory to run the scripts)*

## Installation

To run this project, you need Python and the necessary dependencies. We recommend using the provided Python virtual environment.

1. Clone the repository
2. Activate the virtual environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\\Scripts\\activate`
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

You can explore the system via the provided Jupyter Notebook (`RS Project.ipynb`) which contains the full exploratory data analysis (EDA) and model training code. 

For a streamlined, production-ready script, run the `clean_recommender.py` (if generated) which demonstrates the core recommendation engines.

## Methodology

### 1. Data Cleaning & EDA
- Visualizes course difficulty distributions and top universities using `matplotlib` and `seaborn`.
- Cleans and consolidates text fields (`Course Name`, `Difficulty`, `Skills`) into a single `tags` column for NLP processing.

### 2. NLP Content-Based Approach
- Applies Stemming (NLTK) and `CountVectorizer` to convert textual tags into numeric vectors.
- Computes `cosine_similarity` to find nearest neighbor courses.

### 3. Item-to-Item Collaborative Filtering
- Identifies "similar" courses primarily based on their quantitative metadata (Course Rating, Difficulty Level, and University).
- Converts categorical variables to numeric using `LabelEncoder`, normalizes using `StandardScaler`, and applies Cosine Similarity.

### 4. Deep Learning (Autoencoder Architecture)
- Unsupervised neural network compresses multi-dimensional features down into a 2-Dimensional embedding space.
- The model is trained to minimize the Mean Squared Error (MSE) between the original input features and the reconstructed output for 50 epochs.
- The low-dimensional embeddings are used to generate robust final recommendations.
