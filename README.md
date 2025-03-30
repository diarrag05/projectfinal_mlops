
# Projet MLOps - Détection de Sentiments avec Docker et GitHub Actions

## Description

Ce projet a pour but d'implémenter une solution de détection de sentiments à l'aide de **transformers** (modèle DistilBERT), d'automatiser le pipeline avec **GitHub Actions** et de déployer l'application via **Docker**. Le modèle permet de prédire les sentiments dans des textes en temps réel, tout en étant intégré dans un pipeline MLOps complet, avec tests, évaluation et déploiement.

## Architecture MLOps

Le projet est structuré autour de plusieurs composants clés :

1. **Modèle de détection de sentiments (DistilBERT)** :
   - Utilisation du modèle pré-entrainé **DistilBERT** pour la classification de sentiments dans des textes.
   - Fine-tuning du modèle sur un jeu de données spécifique pour améliorer les performances.

2. **Pipeline CI/CD avec GitHub Actions** :
   - **Workflow de test** : Exécution des tests unitaires et d'intégration à chaque modification (push, pull request).
   - **Workflow d'évaluation** : Evaluation des performances du modèle sur un jeu de données de test après chaque validation.
   - **Workflow de build Docker** : Création et déploiement d'une image Docker.
   - **Workflow de release** : Automatisation des releases et des notes de version.

3. **Déploiement via Docker** :
   - Création d'un **Dockerfile** pour containeriser le modèle.
   - Utilisation de **Docker Compose** pour la gestion des services et des volumes.

## Fonctionnalités

- Prédiction des sentiments dans les textes : positif, négatif, neutre.
- API pour soumettre des textes et recevoir une prédiction en temps réel (via Flask ou FastAPI).
- Automatisation des tests et de l’évaluation du modèle via GitHub Actions.
- Déploiement de l’application avec Docker pour la rendre facilement exécutable sur toute machine.

## Prérequis

- Python 3.8+
- Docker
- Git
- GitHub Account
- Accès à GitHub pour pouvoir créer des pull requests

## Installation

1. Clonez ce repository :
   ```bash
   git clone https://github.com/<votre-utilisateur>/projectfinal_mlops.git
