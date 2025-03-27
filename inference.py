from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

# Charger le modèle
model = AutoModelForSequenceClassification.from_pretrained("results/saved_model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Labels
sentiment_labels = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    return {"text": input.text, "sentiment": sentiment_labels[prediction]}
