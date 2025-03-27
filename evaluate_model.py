import os
from src.data_extraction import load_data
from src.data_processing import preprocess_data
from src.model import train_model
from transformers import AutoModelForSequenceClassification
import evaluate
import numpy as np

# Charger le modèle
model_path = "results/saved_model"
if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Le modèle n'a pas encore été entraîné.")

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Charger les données et prétraiter
data = load_data("dataset.csv")
_, val_dataset = preprocess_data(data)

# Évaluer avec métrique Hugging Face
accuracy = evaluate.load("accuracy")

def compute_accuracy(model, dataset):
    preds = []
    labels = []

    for batch in dataset:
        with torch.no_grad():
            output = model(
                input_ids=batch['input_ids'].unsqueeze(0),
                attention_mask=batch['attention_mask'].unsqueeze(0)
            )
        logits = output.logits
        preds.append(logits.argmax().item())
        labels.append(batch['labels'].item())

    result = accuracy.compute(predictions=preds, references=labels)
    return result['accuracy']

score = compute_accuracy(model, val_dataset)
print(f"✅ Accuracy du modèle : {score:.4f}")

# Définir un seuil minimum acceptable
SEUIL = 0.6
if score < SEUIL:
    raise ValueError(f"❌ Accuracy trop faible : {score:.2f} < {SEUIL}")
