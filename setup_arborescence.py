import os

# Dossiers à créer
folders = [
    "src",
    "tests/unit",
    ".github/workflows",
    "results/saved_model"
]

# Fichiers vides à créer
files = {
    "requirements.txt": "",
    ".gitignore": "",
    "README.md": "# Sentiment BERT MLOps Project\n",
    "train_pipeline.py": "",
    "inference.py": "",
    "Dockerfile": "",
    "docker-compose.yml": "",
    "dataset.csv": "",  # tu peux écraser ce fichier avec ton vrai CSV plus tard
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file_path, content in files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Arborescence du projet MLOps créée avec succès !")
