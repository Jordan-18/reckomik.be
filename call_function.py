# Import Library
import pandas as pd
import numpy as np
import json

# Description Similarity System
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from fuzzywuzzy import process
import imagehash

# description similarity
# Function to get similarity
def get_similarity_description(title, df, cosine_sim):
    if title not in df['title'].values:
        matches = process.extract(title, df['title'], limit=5)
        best_match = matches[0][0]
        print(f"Exact title not found. Using closest match: '{best_match}'")
        title = best_match
        
    idx = df[df['title'] == title].index[0]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = [score for score in sim_scores if score[1] < 1]

    # sim_scores = sim_scores[:100]

    article_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    valid_indices = [i for i in article_indices if i < len(df)]
    
    # Handle cases where valid_indices and scores lengths do not match
    if len(valid_indices) != len(scores):
        min_length = min(len(valid_indices), len(scores))
        valid_indices = valid_indices[:min_length]
        scores = scores[:min_length]
    
    similarity_df = pd.DataFrame({'id': df.iloc[valid_indices]['id'], 'description_similarity': scores})
    result_df = pd.merge(similarity_df, df, on='id')
    
    return result_df

# genre similarity
def matching_genres(df, title):
    if title not in df['title'].values:
        matches = process.extract(title, df['title'], limit=5)
        best_match = matches[0][0]
        print(f"Exact title not found. Using closest match: '{best_match}'")
        title = best_match
        
    data = df[df['title'] == title]
    main_genres = (list(data.head(1)['genre'])[0]).split(', ')
    genre_similarity = df['genre']
    
    for i, rec in enumerate(genre_similarity):
        match_count = 0
        if isinstance(rec, float): 
            rec = str(rec) 
        genres = rec.split(', ')
        for j, genre in enumerate(genres):
            if genre in main_genres:
                match_count += 1
        
        genre_similarity = (match_count / len(main_genres)) * 100
        df.at[i, 'genre_similarity'] = genre_similarity
    filtered_df = df.copy()
        
    return filtered_df

# image similarity
def compute_image_similarity(hash1, hash2):
    return 1 - (hash1 - hash2) / len(hash1.hash) ** 2

def get_similarity_image(title, df):
    if title not in df['title'].values:
        matches = process.extract(title, df['title'], limit=5)
        best_match = matches[0][0]
        print(f"Exact title not found. Using closest match: '{best_match}'")
        title = best_match
        
    target_row = df[df['title'] == title].iloc[0]
    target_hash_str = target_row['image_hash']
    
    if not isinstance(target_hash_str, str):
        print("Target image hash is not valid.")
        return df
    
    target_hash = imagehash.hex_to_hash(target_hash_str)
    
    similarities = []
    for index, row in df.iterrows():
        if row['title'] != title:
            other_hash_str = row['image_hash']
            if not isinstance(other_hash_str, str):
                continue
            try:
                other_hash = imagehash.hex_to_hash(other_hash_str)
                similarity = compute_image_similarity(target_hash, other_hash)
                similarities.append((row['id'], similarity))
            except Exception as e:
                print(f"Error processing image hash for title '{row['title']}': {e}")
                continue
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    similarity_df = pd.DataFrame(similarities, columns=['id', 'image_similarity'])
    result_df = pd.merge(similarity_df, df, on='id')
    
    return result_df

def all_similarity(title, df, cosine_sim):
    columns_to_drop = ['image_similarity','genre_similarity', 'description_similarity']
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    if existing_columns:
        df = df.drop(existing_columns, axis=1)

    df = matching_genres(df, title) # Genre Similarity Function
    df = get_similarity_description(title, df, cosine_sim) # Description Similarity Function
    df = get_similarity_image(title, df) # Image Similarity Function
    
    df = df.sort_values(by=existing_columns, ascending=False)

    return df