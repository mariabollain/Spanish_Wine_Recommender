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

notes = wine_clusters["notes_norm_removed_new_reduction_dropped"].str.split()
w2v_model, tfidf_vectorizer, tfidf_matrix = train_w2v_and_tfidf_models(notes)

# Train the KMeans model
embeddings_array = np.array(wine_clusters["embeddings"].tolist())
@st.cache_resource
def train_kmeans_model(embeddings_array):
    kmeans = KMeans(n_clusters=13, random_state=42)
    kmeans.fit(embeddings_array)
    return kmeans

kmeans = train_kmeans_model(embeddings_array)

# Function to obtain recommendation
def find_similar_wines(wine_name, filters, num_similar=5):
    # Get the embedding and the cluster of the input wine
    embedding = np.array(wine_clusters.loc[wine_clusters["wine"] == wine_name, "embeddings"].values[0])
    cluster = wine_clusters.loc[wine_clusters["wine"] == wine_name, "cluster13"].values[0]

    # Obtain the wines from the same cluster and apply user filters
    cluster_wines = wine_clusters[wine_clusters["cluster13"] == cluster]

    type_mask = cluster_wines["type_wine"] == filters["type"]
    if filters["region"] == "All":
        region_mask = np.ones(len(cluster_wines), dtype=bool)  # Select all wines (no filtering by region)
    else:
        region_mask = cluster_wines["region_gi"] == filters["region"]
    if filters["grapes"] == ["All"]:
        grapes_mask = np.ones(len(cluster_wines), dtype=bool)  # Select all wines (no filtering by grapes)
    else:
        if "All" in filters["grapes"]:
            filters["grapes"].remove("All")
        grapes_mask = cluster_wines["grapes"].apply(lambda x: any(grape in x for grape in filters["grapes"]))
    price_mask = cluster_wines["price"].between(*filters["price"])
    year_mask = cluster_wines["year"].between(*filters["year"])
    reviews_mask = cluster_wines["customer_reviews"].between(*filters["reviews"])

    filtered_wines = cluster_wines[type_mask & region_mask & grapes_mask & price_mask & year_mask & reviews_mask]

    # Check if the filtered_wines dataframe is empty
    if filtered_wines.empty:
        st.write("No wines found matching the specified criteria.")
        return []
    else:
        # Calculate cosine similarity between the input wine and all wines in the same cluster
        cosine_similarities = cosine_similarity(np.vstack(filtered_wines["embeddings"]), embedding.reshape(1, -1)) # vstack and reshape turn into 2D arrays to ensure compatibility with cosine_similarity

        # Get the indices of the most similar wines in the cluster
        similar_wine_indices = np.argsort(cosine_similarities.flatten())[:num_similar + 1]  # +1 to include the input wine itself

        # Get the index of the input wine
        input_wine_index = wine_clusters.index[wine_clusters["wine"] == wine_name].tolist()[0]

        # Exclude the input wine from the list of similar wine indices
        similar_wine_indices = similar_wine_indices[similar_wine_indices != input_wine_index]

        # Get the names of the most similar wines (excluding the input wine)
        similar_wines = filtered_wines.iloc[similar_wine_indices]["wine"].tolist()

    return similar_wines

# Defining main function
def main():

    # Title
    st.title("Wine recommender")

    # Main user input: wine name
    user_wine = st.selectbox(label="Please, select a wine", options=wine_clusters)
    wine_info = wine_clusters[wine_clusters["wine"] == user_wine].iloc[0]
    wine_type = wine_info["type_wine"]
    wine_link = wine_info["url"]
    wine_image = wine_info["image"]

    st.image(wine_image, width=250) #125
    st.caption(f"[{user_wine}]({wine_link})")

    # Optional user inputs in sidebar
    st.sidebar.header("Optional filter parameters")

    region_options = ['All', 'Rioja', 'Ribera del Duero', 'DO Cava', 'Penedès',
                      'Wines without GI', 'Rías Baixas', 'Priorat', 'Rueda', 'Montsant',
                      'Empordà', 'Castile and León', 'Bierzo', 'Terra Alta', 'Toro',
                      'Somontano', 'Navarre', 'Catalunya', 'Jumilla', 'Castilla',
                      'Costers del Segre', 'Jerez-Manzanilla', 'Corpinnat',
                      'Ribeiro', 'Alicante', 'Carignan', 'Campo de Borja',
                      'Conca de Barberà', 'Valencia', 'Ribeira Sacra',
                      'Mallorca', 'Clàssic Penedès', 'Montilla-Moriles', 'Pla de Bages',
                      'Valdeorras', 'Yecla', 'Cádiz', 'Vinos de Madrid', 'Calatayud',
                      'Tarragona', 'Monterrei', 'Alella', 'Cigales', 'La Mancha',
                      'Utiel-Requena', 'Sierras de Málaga', 'Valdejalón', 'Méntrida',
                      'Extremadura', 'Lanzarote', 'Binissalem-Mallorca', 'Abona', 'Ibiza',
                      'Pla i Llevant', 'Tierra de León', 'Arlanza', 'Tacoronte - Acentejo',
                      'Ycoden-Daute-Isora', 'Granada', 'Almansa', 'Sierra de Salamanca',
                      'Bizkaiko Txakolina', 'Getariako Txakolina', 'Valles de Sadacia',
                      'Canary Islands', 'Conca del Riu Anoia', 'Valle de Güímar',
                      'Abadía Retuerta', 'Cebreros', 'Bullas', 'Illa de Menorca',
                      'Valle de La Orotava', 'Manchuela', 'Málaga', 'Formentera', 'Arribes',
                      '3 Riberas', 'Condado de Huelva', 'Bajo Aragón', 'Ribera del Queiles',
                      'La Palma', 'Ribera del Guadiana', 'Barbanza e Iria', 'Cangas',
                      'Dominio de Valdepusa', 'Vallegarcía', 'Tierra del Vino de Zamora', 'Uclés']

    region = st.sidebar.selectbox(label="Geographical indication", options=region_options)

    grapes_options = ['All', 'Tempranillo', 'Garnacha', 'Cabernet Sauvignon', 'Syrah', 'Macabeo',
       'Xarel·lo', 'Merlot', 'Cariñena', 'Chardonnay', 'Parellada', 'Albariño',
       'White Grenache', 'Verdejo', 'Graciano', 'Monastrell', 'Mencia',
       'Viura', 'Pinot Noir', 'Muscat of Alexandria', 'Godello', 'Mazuelo',
       'Palomino Fino', 'Malvasia', 'Sauvignon Blanc', 'Pedro Ximénez',
       'Tinta de Toro', 'Treixadura', 'Garnacha Tintorera', 'Albillo',
       'Sumoll', 'Bobal', 'Trepat', 'Cabernet Franc', 'Loureiro',
       'Listán Negro', 'Petit Verdot', 'Mantonegro',
       'Muscat Blanc à Petits Grains', 'Callet', 'Gewürztraminer', 'Viognier',
       'Samsó', 'Sousón', 'Caiño Tinto', 'Listán Blanco'] # Grapes that appear in at least 10 wines

    grapes = st.sidebar.multiselect("Grape variety", grapes_options, ["All"])

    price_range = st.sidebar.select_slider(
        "Price",
        options=list(range(0, 201)),
        value=(5, 40))

    year_range = st.sidebar.select_slider(
        "Year",
        options=list(range(2000,2024)),
        value=(2017, 2023))

    reviews_range = st.sidebar.select_slider(
        "Reviews",
        options=list(np.arange(0.0, 5.5, 0.5)),
        value=(4.0, 5.0))

    filters = {"type": wine_type,
               "region": region,
               "grapes": grapes,
               "price": price_range,
               "year": year_range,
               "reviews": reviews_range}

    # Button for running the recommender
    if st.button("RECOMMEND"):
        st.subheader("Recommendations")
        recommendations = find_similar_wines(user_wine, filters)
        if len(recommendations) > 5:
            recommendations = recommendations[:5]
        col1, col2, col3, col4, col5 = st.columns(5)
        i = 1
        for wine in recommendations:
            with locals()[f"col{i}"]:
                wine_info = wine_clusters[wine_clusters["wine"] == wine].iloc[0]
                wine_name = wine_info["wine"]
                wine_link = wine_info["url"]
                wine_image = wine_info["image"]
                st.image(wine_image, use_column_width="always")
                st.caption(f"[{wine_name}]({wine_link})")
                if i < 5:
                    i += 1


if __name__ == '__main__':
    main()