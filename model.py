import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

ratings = pd.read_csv("/media/manvendra07/Manvendra's Passport/Documents/ML-Project-RecommenderSystem/ratings.csv")
movies = pd.read_csv("/media/manvendra07/Manvendra's Passport/Documents/ML-Project-RecommenderSystem/movies.csv")
links = pd.read_csv("/media/manvendra07/Manvendra's Passport/Documents/ML-Project-RecommenderSystem/links.csv")
tags = pd.read_csv("/media/manvendra07/Manvendra's Passport/Documents/ML-Project-RecommenderSystem/tags.csv")
df = pd.read_csv("/media/manvendra07/Manvendra's Passport/Documents/ML-Project-RecommenderSystem/pivot.csv")
df.rename(columns={'Unnamed: 0' : 'userId'}, inplace=True)

csr_data = csr_matrix(df.values)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = df[df['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = df.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append(str(movies.iloc[idx]['title'].values[0]).zfill(7))
        return recommend_frame
    else:
        return "No movies found. Please check your input"
