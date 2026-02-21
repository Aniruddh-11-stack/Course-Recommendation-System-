import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Download NLTK data (if not already downloaded in the environment)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import os

print("Loading dataset...")
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'Coursera.csv')
    data = pd.read_csv(csv_path)
    print(f"Dataset loaded. Shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'Coursera.csv' not found. Please ensure the dataset is in the same directory.")
    exit(1)

# --- 1. Content-Based Recommendation (NLP approach) ---
print("\n--- Initializing Content-Based Recommender ---")

# Data preprocessing specifically for text similarity
new_data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']].copy()

# Clean spaces and symbols
for col in ['Course Name', 'Course Description', 'Skills']:
    if col in new_data.columns:
        new_data[col] = new_data[col].astype(str).str.replace(r'[\(\):_]', '', regex=True)
        new_data[col] = new_data[col].str.replace(' ', ',')
        new_data[col] = new_data[col].str.replace(',,', ',')

# Create combined tags
new_data['tags'] = new_data['Course Name'] + " " + new_data['Difficulty Level'] + " " + new_data['Course Description'] + " " + new_data['Skills']
new_df = new_data[['Course Name', 'tags']].copy()
new_df.rename(columns={'Course Name': 'course_name'}, inplace=True)
new_df['tags'] = new_df['tags'].str.replace(',', ' ').str.lower()
new_df['course_name'] = new_df['course_name'].str.replace(',', ' ')

# Apply stemming
ps = PorterStemmer()
def stem(text):
    return " ".join([ps.stem(word) for word in text.split()])

try:
    new_df['tags'] = new_df['tags'].apply(stem)
except Exception as e:
    print(f"Warning: Stemming failed, proceeding without it. {e}")

# Vectorization and similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend_similar_course(course_title, num_recommendations=5):
    """Recommend similar courses based on content tags."""
    try:
        # Find index of the course
        course_index = new_df[new_df['course_name'] == course_title].index[0]
        distances = similarity[course_index]
        # Sort by similarity score
        course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:num_recommendations+1]
        
        print(f"\nRecommendations for '{course_title}':")
        for i in course_list:
            print(f"- {new_df.iloc[i[0]].course_name}")
    except IndexError:
        print(f"Course '{course_title}' not found in the dataset.")

# Interactive Terminal Chat
print("\n" + "*"*60)
print("Welcome to the Course Recommendation System!")
print("Type 'quit' or 'exit' at any time to leave.")
print("*"*60)

while True:
    try:
        user_input = input("\nEnter a course name you like (or 'quit'): ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Try to find an exact match first
        if user_input in new_df['course_name'].values:
            recommend_similar_course(user_input)
        else:
            # Try to find a partial case-insensitive match
            matches = new_df[new_df['course_name'].str.contains(user_input, case=False, na=False)]
            
            if len(matches) == 0:
                print(f"Sorry, I couldn't find any course matching '{user_input}'. Try another term.")
            elif len(matches) == 1:
                matched_course = matches.iloc[0]['course_name']
                print(f"Found match: '{matched_course}'")
                recommend_similar_course(matched_course)
            else:
                print(f"Found multiple matches for '{user_input}'. Please be more specific. Top matches:")
                for c in matches['course_name'].head(5):
                    print(f"- {c}")
                    
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
