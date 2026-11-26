# MNIST et RotNet avec 100 labels

Projet de Deep Learning — M2 MLSD  
**Équipe :** Sabrina Belaribi, Chaimaa Arsalane, Sirine Boughanem et Manel Lakrouz.

## Objectif

Étudier si un préentraînement auto-supervisé par prédiction de rotations (RotNet) peut améliorer la classification des chiffres MNIST lorsque seuls 100 labels de chiffres sont disponibles pour l'apprentissage.

L'expérience A présentée dans le notebook est une **référence supervisée** entraînée avec les 40 000 labels. Elle sert de comparaison et ne respecte pas la contrainte des 100 labels. Ses poids ne doivent pas servir à préentraîner les méthodes à 100 labels.

## Préparation

Avec Python 3.13, installer les dépendances depuis le terminal à la racine du projet :

```powershell
python -m pip install -r requirements.txt
```

Ouvrir ensuite `pipeline.ipynb` dans VS Code et sélectionner l'environnement Python. Le premier chargement télécharge MNIST dans `data/` si les fichiers sont absents.

## Fichiers principaux

- `pipeline.ipynb` : exécution, visualisations et résultats de la référence A.
- `src/data.py` : chargement des données et préparation du pool commun.
- `src/models.py` : architecture CNN commune.
- `src/engine.py` : fonctions d'entraînement et d'évaluation.
- `src/train_supervised.py` : entraînement et sauvegarde de la référence A.
- `src/transfer_learning.py` : fichier prévu pour le transfert.
- `splits/pool_seed42.npz` : indices du pool commun de 40 000 images, dans le jeu d'entraînement officiel MNIST.
- `results/reference_A/` : métriques et poids de la référence A.

Le pool est un tirage proposé en l'absence d'un découpage fourni par l'enseignant. Les 10 000 images de test officielles sont réservées à l'évaluation. Un nouvel entraînement relance les 30 époques et doit utiliser un dossier de résultats qui n'existe pas encore : le programme refuse de remplacer un dossier existant.

## Article de référence

Gidaris, S., Singh, P. et Komodakis, N. (2018). *Unsupervised Representation Learning by Predicting Image Rotations*. ICLR.
