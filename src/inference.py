from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("results/saved_model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = logits.argmax().item()
    labels = ["Très négatif", "Négatif", "Neutre", "Positif", "Très positif"]
    return labels[pred]

if __name__ == "__main__":
    example = "I love using this app!"
    print(f"Texte : {example}")
    print(f"Prédiction : {predict(example)}")
