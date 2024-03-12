import base64
import json
import os
import time
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from sklearn.cluster import KMeans
import requests
import numpy as np
import tqdm

OPENAI_KEY = os.environ.get('OPENAI_KEY')
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET')

if not OPENAI_KEY or not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    raise Exception('Environment variables OPENAI_KEY, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET are required')


app = FastAPI()
openai = OpenAI(api_key=OPENAI_KEY)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

existing_embeddings_fn = 'existing_embeddings.json'
existing_embeddings = {}
with open(existing_embeddings_fn, 'r') as f:
    existing_embeddings = json.load(f)
    print(f'load {len(existing_embeddings)} embeddings')

def embedd_genres(genres, existing_embeddings, model='text-embedding-3-small'):
    total_tokens = 0
    embeddings = {}
    print('loading embeddings')
    for i, genre in tqdm.tqdm(enumerate(genres)):
      if genre in existing_embeddings:
        embeddings[genre] = existing_embeddings[genre]
        continue
      response = openai.embeddings.create(input=genre, model=model)
      embeddings[genre] = response.data[0].embedding
      total_tokens += response.usage.total_tokens
    return embeddings, total_tokens

class ClusterRequest(BaseModel):
    genres: List[str]


class InitTokenRequest(BaseModel):
    code: str
    redirect_uri: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

@app.post("/api/clusters")
def generate_clusters(request: ClusterRequest):
    if len(request.genres) < 10:
        return { 'error': 'At least 10 genres are required' }
    embeddings, total_tokens = embedd_genres(request.genres, existing_embeddings)    
    update_existing_embeddings = {**existing_embeddings, **embeddings}
    with open(existing_embeddings_fn, 'w') as f:
        json.dump(update_existing_embeddings, f)

    X = np.array(list(embeddings.values()))
    kmeans = KMeans(n_clusters=7, random_state=0).fit(X)

    genre_clusters = {}
    for genre, embedding in embeddings.items():
        cluster = kmeans.predict([embedding])[0]
        cluster = f"cluster_{cluster}"
        if cluster not in genre_clusters:
            genre_clusters[cluster] = []
        genre_clusters[cluster].append(genre)
    return { 'clusters': genre_clusters, 'total_tokens': total_tokens}


@app.post("/api/token")
def init_token(request: InitTokenRequest):
    url = 'https://accounts.spotify.com/api/token'

    encoded_auth_token = base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode('utf-8')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Auuthorization': f'Basic {encoded_auth_token}'
    }
    body = {
        'grant_type': 'authorization_code',
        'code': request.code,
        'redirect_uri': request.redirect_uri
    }

    response = requests.post(url, headers=headers, data=body)
    json_response = response.json()
    print(json_response)

    expires_timestamp_in_seconds = json_response['expires_in'] + int(time.time())

    return {
        'access_token': json_response['access_token'],
        'refresh_token': json_response['refresh_token'],
        'expires_at': expires_timestamp_in_seconds
    }


@app.post("/api/refresh-token")
def get_token(request: RefreshTokenRequest):
    url = 'https://accounts.spotify.com/api/token'

    encoded_auth_token = base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode('utf-8')
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Auuthorization': f'Basic {encoded_auth_token}'
    }
    body = {
        'grant_type': 'refresh_token',
        'refresh_token': request.refresh_token
    }

    response = requests.post(url, headers=headers, data=body)
    json_response = response.json()
    expires_timestamp_in_seconds = json_response['expires_in'] + int(time.time())

    return {
        'access_token': json_response['access_token'],
        'refresh_token': request.refresh_token,
        'expires_at': expires_timestamp_in_seconds
    }

@app.get("/test")
def test():
    return { 'test': 'test' }