import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Get the list of stop words in English
stop_words = stopwords.words('english')

# Custom Transformer for Text Preprocessing
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words):
        self.stop_words = stop_words
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.apply(self._preprocess_text)
    
    def _preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(filtered_tokens)

# Function to prepare the pipeline and cosine similarity matrix
def prepare_similarity_system(df):
    # Pipeline for TF-IDF Vectorization with Preprocessing
    pipeline = Pipeline([
        ('preprocessor', TextPreprocessor(stop_words)),
        ('tfidf', TfidfVectorizer())
    ])

    # Apply pipeline to descriptions
    tfidf_matrix = pipeline.fit_transform(df['description'].astype(str))

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

# Function to get similarity
def get_similarity(title, df, cosine_sim):
    # Get the index of the given title
    idx = df[df['title'] == title].index[0]
    
    # Sort the similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the top 20 most similar articles
    sim_scores = sim_scores[1:21]
    
    # Get the indices of these articles and their scores
    article_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    # Return the titles of the recommended articles and their scores
    similarity = [({
        'title': df['title'].iloc[i],
        'alt_title': df['alt_title'].iloc[i],
        'type': df['type'].iloc[i],
        'description': df['description'].iloc[i],
        'genre': df['genre'].iloc[i],
        'author': df['author'].iloc[i],
        'artist': df['artist'].iloc[i],
        'rate': df['rate'].iloc[i],
        'image': df['image'].iloc[i],
        'released': df['released'].iloc[i],
        'description_similiraty': scores[idx]
    }) for idx, i in enumerate(article_indices)]
    
    return similarity
