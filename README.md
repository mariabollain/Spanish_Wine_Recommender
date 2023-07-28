# Final Project: Spanish Wines Recommender
The aim of this project is to build a wine recommender using unsupervised machine learning techniques. 

Most online wine shops tend to recommend wines that are very similar to the initial one (usually, they are wines from the same region or even from the same winery). 

The aim with this project is to recommend to customers wines in the style they enjoy, but not necessarily from the same region or grape variety.  Wines that can surprise them by discovering something new.

## Tools and links to files
- Python (Jupiter Notebooks)
     - [Webscrapping](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/01-initial_data/webscraping/webscraping_clean.ipynb)
     - [Cleaning](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/02-cleaning/cleaning.ipynb)
     - [Exploratory Data Analysis](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/03-eda/eda.ipynb)
     - [Machine Learning: Clustering](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/04-clustering/clustering.ipynb)
     - [Building the recommender](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/04-clustering/recommender.ipynb)
-  Python (Streamlit app)
     -  [Script](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/05-app/main.py)
 - [Tableau](https://public.tableau.com/views/wine_dashboard_16904972267650/Dashboard12?:language=es-ES&:display_count=n&:origin=viz_share_link)
 - [SQL](https://github.com/mariabollain/Final_project_Wine_recommender/blob/main/06-sql/SQL-queries.sql)
 - Slides    

## Process
To collect the data I have webscraped Vinissimus.com, an online shop specialised in Spanish wines, to obtain a dataset with information about 5345 unique wines. 

After cleaning the dataset, I carried out an exhaustive exploratory analysis of the different features, with special emphasis on the insights we can get from customer reviews and tasting notes (bouquet and mouth). These tasting notes are the only features I have used for clustering, to find wines that are similar in bouquet and mouth, regardless of other characteristics.

The biggest difficulty in the project has been the enormous variability in tasting descriptors, with 653 different terms, which on average appeared in less than 2% of the wines in the dataset. So I carried out a task of simplifying the tasting terms, merging them into broader categories (e.g. "strawberry" and "raspberry" into "red fruit"), followed by the elimination of all those terms appearing in less than 300 wines.

I then applied Word2Vec to obtain embeddings from the different terms, and TF-IDF to obtain vectors for each wine. Here we encounter the second difficulty in the project, and that is the limited size of the training data. Since the corpus used for training the model was not large enough, the quality of the word embeddings was not very high.
Finally, I carried out clustering using different models (KMeans, DBSCAN, Gaussian Mixture).

From the input (a wine from the dataset), the recommender selects the five most similar wines within the same cluster, using cosine similarity.

## Results
![result-TruncatedSVD](https://github.com/mariabollain/Final_project_Wine_recommender/assets/122167121/fdff99ce-232b-4351-aba9-492a1d05d343)
Clustering with KMeans and K=13, after dimensionality reduction with TruncatedSVD.



---
This project was developed in three weeks as the final project for the Data Analytics bootcamp at Ironhack.
