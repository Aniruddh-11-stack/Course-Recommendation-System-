### Summary of Item-Based Collaborative Filtering Code

**Item-based collaborative filtering** is a recommendation algorithm that suggests products based on item similarities. Here's a summary of how it works and the code provided:

#### 1. Introduction
- **Item-based collaborative filtering** uses item similarity for recommendations.
- Assumes users rate similar products similarly.

#### 2. Algorithm Steps
1. **Calculate item similarity scores** from user ratings.
2. **Identify top n similar items** to the target item.
3. **Calculate weighted average scores** for similar items rated by the user.
4. **Rank items** and recommend the top n items.

#### 3. Code Walkthrough

**Import Libraries:**
- Data processing: `pandas`, `numpy`, `scipy.stats`.
- Visualization: `seaborn`.
- Similarity calculation: `cosine_similarity` from `sklearn`.

**Data Preparation:**
- Load and explore the `ratings.csv` and `movies.csv` files.
- Merge datasets on `movieId`.

**Exploratory Data Analysis (EDA):**
- Aggregate ratings by movie.
- Filter movies with over 100 ratings.
- Merge datasets to retain movies with sufficient ratings.

**Create User-Movie Matrix:**
- Transform dataset into a matrix with movies as rows and users as columns.
- Normalize data by subtracting the average rating for each movie.

**Calculate Similarity Scores:**
- Use Pearson correlation and cosine similarity to compute item similarity matrices.

**Predict User's Rating for a Movie:**
1. Select a user and a movie.
2. Find movies rated by the user.
3. Rank similarity scores of rated movies to the target movie.
4. Calculate predicted rating using weighted average of similarities and ratings.

**Movie Recommendation:**
- For a target user, predict scores for unwatched movies.
- Rank predicted scores and recommend top movies.

**Python Function for Recommendations:**
- Input: `picked_userid`, `number_of_similar_items`, `number_of_recommendations`.
- Output: Top recommended movies and their predicted ratings.

**Example Implementation:**
- Predict rating for user 1 and movie "American Pie (1999)".
- Print recommended movies for a user.

#### 4. Detailed Code Examples

**Sample Data:**
```python
web_series_ratings = {
    'WS1': {'2': 4, '3': 3, '4': 4, '5': 3, '6': 5},
    'WS2': {'1': 5, '2': 3, '3': 4, '4': 2, '5': 4, '6': 5},
    # More web series ratings
}
data = pd.DataFrame.from_dict(web_series_ratings)
```

**Unique Users and Series Functions:**
```python
def uniqueUser(ratings):
    users = set(user for rating in ratings.values() for user in rating.keys())
    print(users)

def uniqueSeries(ratings):
    series = set(ratings.keys())
    print(series)
```

**Cosine Similarity Calculation:**
```python
def cosine_similarity(item1, item2):
    items = pd.concat([item1, item2], axis=1).dropna()
    item1, item2 = items[items.columns[0]], items[items.columns[1]]
    dot_product = np.sum(item1 * item2)
    mag1, mag2 = np.sqrt(np.sum(np.square(item1))), np.sqrt(np.sum(np.square(item2)))
    return dot_product / (mag1 * mag2)
```

**Find Similarity for All Items:**
```python
def findAllSim(ratings, target):
    sim = {i: cosine_similarity(ratings[target], ratings[i]) if i != target else -1 for i in ratings.columns}
    return pd.DataFrame.from_dict(sim)
```

**Predict Rating:**
```python
def rating(k, df, item, user):
    df_mean = df.subtract(df.mean(axis=1), axis='rows')
    sim = findAllSim(df_mean, item)
    top_columns = sim.iloc[0].nlargest(k).index
    rating = sum(sim[i] * df[i][user] for i in top_columns) / np.sum(sim[top_columns], axis=1)
    return rating
```

#### 5. Conclusion

The code demonstrates item-based collaborative filtering using both Pearson correlation and cosine similarity to predict user ratings and recommend movies. The process involves calculating item similarities, predicting ratings for a specific user and movie, and generating movie recommendations based on those predictions.
