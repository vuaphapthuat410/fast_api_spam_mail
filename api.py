from typing import List

import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from joblib import load

from fastapi import Depends, FastAPI

app = FastAPI()

clf = load('spam_mail_multiNB.joblib') 

count_vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

@app.post("/predict")
async def predict(text: str):
	count_vec = count_vectorizer.transform([text])
	pred = clf.predict(count_vec)
	return {"result": int(pred[0])}

@app.get("/raw_predict")
async def raw_predict(text: str):
	count_vec = count_vectorizer.transform([text])
	pred = clf.predict(count_vec)
	return {"result": int(pred[0])}
