# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import multiprocessing
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from ast import literal_eval

warnings.filterwarnings('ignore')

# Load the data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, converters={"embeddings": literal_eval})
    return df

path = "C:/Users/mboll/OneDrive/Documentos/DATA/Ironhack/Final_project/04-clustering/wine_clusters.csv"
wine_clusters = load_data(path)

# Define the custom tokenizer function at the module level to avoid errors
def custom_tokenizer(text):
    return text.split()

# Train the Word2Vec and TF-IDF models
cores = multiprocessing.cpu_count()
@st.cache_data
def train_w2v_and_tfidf_models(notes):
    # Train Word2Vec
    w2v_model = Word2Vec(min_count=70,
                         window=2,
                         vector_size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         workers=cores - 1)
    w2v_model.build_vocab(notes, progress_per=10000)
    w2v_model.train(notes, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    # Train TF-IDF
    tfidf_vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer)
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(note) for note in notes])

    return w2v_model, tfidf_vectorizer, tfidf_matrix

notes = wine_clusters["notes_norm_removed_reduced250"].str.split()
w2v_model, tfidf_vectorizer, tfidf_matrix = train_w2v_and_tfidf_models(notes)

# Train the KMeans model
embeddings_array = np.array(wine_clusters["embeddings"].tolist())
@st.cache_resource
def train_kmeans_model(embeddings_array):
    kmeans = KMeans(n_clusters=13, random_state=42)
    kmeans.fit(embeddings_array)
    return kmeans

kmeans = train_kmeans_model(embeddings_array)

# Function to obtain the embedding from the text input
def text_to_vector(text_input, w2v_model, tfidf_vectorizer, tfidf_matrix):
    # Tokenize the text input using the custom tokenizer
    words = custom_tokenizer(text_input)

    # Convert words to Word2Vec vectors
    word_vectors = []
    valid_words = []
    for word in words:
        if word in w2v_model.wv:
            word_vector = w2v_model.wv[word]
            word_vectors.append(word_vector)
            valid_words.append(word)

    # Get the indices of valid words in the TF-IDF vectorizer's vocabulary
    word_indices = [tfidf_vectorizer.vocabulary_.get(word, -1) for word in valid_words]

    # Get the TF-IDF scores for the valid words from the TF-IDF matrix
    tfidf_scores = tfidf_matrix[:, word_indices].toarray()

    # Calculate the TF-IDF weighted average vector of the input text
    tfidf_avg_vector = np.zeros(w2v_model.vector_size)
    for word, tfidf_score in zip(words, tfidf_scores[0]):
        if word in w2v_model.wv:
            word_vector = w2v_model.wv[word]
            tfidf_avg_vector += word_vector * tfidf_score

    if len(word_vectors) > 0:
        tfidf_avg_vector /= len(word_vectors)

    # Reshape the vector to be a 2D array as expected by Kmeans
    tfidf_avg_vector = tfidf_avg_vector.reshape(1, -1)

    return tfidf_avg_vector


# Function to obtain recommendation
def recommend_from_descriptors(descriptors, num_similar=5):
    # Get the embedding and the cluster of the input
    embedding = text_to_vector(descriptors, w2v_model, tfidf_vectorizer, tfidf_matrix)
    cluster = kmeans.predict(embedding)[0]

    # Filter the dataframe to select only the predicted cluster
    filtered_df = wine_clusters[wine_clusters["clusters"] == cluster]

    # Get five random wines from the filtered dataframe
    similar_wines = filtered_df.sample(5, ignore_index=True)

    return cluster, embedding, similar_wines

# Defining main function
def main():

    # Title
    st.title("Wine recommender")

    # Type of wine
    type = st.radio(
        "What is your favorite type of wine?",
        ("Red", "White", "Rosé", "Sparkling", "Dessert"))

    st.write('You selected:', type)

    st.write("How do you want the wine to be?")

    options = ['black_fruit',
             'wood',
             'elegant',
             'silky',
             'rich',
             'floral_red',
             'charcoal',
             'toasty',
             'liquorice',
             'bakery',
             'spices',
             'nuts',
             'rounded',
             'floral_white',
             'good_acidity',
             'dried_fruit',
             'unctuous',
             'tropical_fruit',
             'ageing',
             'powerful',
             'pleasant',
             'flavoursome',
             'long_finish',
             'red_fruit',
             'white_fruit',
             'balanced',
             'long',
             'mineral',
             'ripe_fruit',
             'mediterranean_herbs',
             'persistent',
             'vanilla',
             'complex',
             'balsamic',
             'fruit-forward',
             'full',
             'fresh',
             'stone_fruit',
             'earthy',
             'crunchy',
             'other_herbs',
             'pleasant_finish',
             'caramel',
             'structured']

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        user_options1 = {word: st.checkbox(f"{word}") for word in options[:10]}

    with col2:
        user_options2 = {word: st.checkbox(f"{word}") for word in options[10:20]}

    with col3:
        user_options3 = {word: st.checkbox(f"{word}") for word in options[20:30]}

    with col4:
        user_options4 = {word: st.checkbox(f"{word}") for word in options[30:]}

    user_options = {}
    user_options.update(user_options1)
    user_options.update(user_options2)
    user_options.update(user_options3)
    user_options.update(user_options4)

    if sum(user_options.values()) > 3:
        st.write('You have selected more than three terms. Only the first three will be considered.')
    else:
        st.write("Please select up to three terms.")

    selected_options = [word for word, checked in user_options.items() if checked]
    selected_options_text = " ".join(selected_options)
    st.write(selected_options_text)

    if len(selected_options)==1:
        st.write(f"You selected: {selected_options[0]}.")
    elif len(selected_options)==2:
        st.write(f"You selected: {selected_options[0]} and {selected_options[1]}.")
    elif len(selected_options)>2:
        st.write(f"You selected: {selected_options[0]}, {selected_options[1]} and {selected_options[2]}.")

    # Create button for running the recommender
    if st.button("RECOMMEND"):
        cluster, embedding, recommendations = recommend_from_descriptors(selected_options_text)
        if np.all(embedding == 0):  # Check if the embedding is all zeros
            st.write("Not able to recommend")
        else:
            # img_url = list(wine_clusters[wine_clusters["wine"]==recommendations[0]]["image"])[0]
            # caption = f"{wine_name}<br>{description}"
            # st.image(image, caption=caption, use_column_width=True)
            st.write(cluster)
            st.dataframe(recommendations)
            # st.image(img_url)

if __name__ == '__main__':
    main()