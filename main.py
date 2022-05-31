from fastapi import FastAPI
import uvicorn
import model
app = FastAPI(
    title="Recommender System API",
    description="A simple API that use KNN model to recommend top 10 movies based on Genre",
    version="0.1",
)

@app.get("/get_recommendations")
async def recommend(movie: str):
    recommendations = model.get_movie_recommendation(movie)
    return recommendations