# GuardVision - Système de Surveillance Intelligent
système de surveillance basé sur la vision par ordinateur et l'intelligence artificielle 

-models link : https://drive.google.com/drive/folders/1kTMjxi66thDcsd1EUR81ZwlWA2Tp2Cms?usp=drive_link
-report link : https://app.readthedocs.org/projects/guardvision2/

## Description du projet

GuardVision est un projet académique de surveillance vidéo intelligente qui utilise l'intelligence artificielle pour détecter automatiquement les activités suspectes et analyser la densité des foules en temps réel.

## Objectifs

- Développer un système de reconnaissance d'actions automatisé
- Implémenter l'analyse de densité de foule
- Créer une interface utilisateur intuitive pour la surveillance
- Démontrer l'application pratique de l'IA dans la sécurité

## Fonctionnalités principales

### Reconnaissance d'actions
Détection automatique de 13 types d'activités :
- Actions normales et suspectes (vol, agression, vandalisme, etc.)
- Situations d'urgence (incendie, explosion)
- Comportements anormaux

### Analyse de foule
- Mesure de la densité en temps réel
- Détection de rassemblements
- Alertes de surcharge

### Détection de pose humaine
- Estimation des points clés du corps
- Analyse des mouvements et postures
- Visualisation du squelette 3D

## Technologies utilisées

- **Intelligence Artificielle** : TensorFlow, MediaPipe
- **Vision par ordinateur** : OpenCV, YOLOv5
- **Interface utilisateur** : Streamlit
- **Traitement d'images** : NumPy, PyTorch

## Installation et utilisation

1. Installer les dépendances Python requises
2. Placer les modèles installés depuis le lien ci-dessus pré-entraînés dans le dossier `models/`
3. Lancer l'application avec Streamlit
4. Configurer les modules de détection via l'interface web

## Architecture

Le système comprend :
- **Module de détection d'actions** avec modèle TensorFlow personnalisé
- **Module d'analyse de foule** pour mesurer la densité
- **Détecteur de pose** utilisant MediaPipe
- **Interface web** pour la surveillance en temps réel
- **Système d'alertes** avec historique des détections

## Résultats attendus

- Démonstration d'un système de surveillance automatisé fonctionnel
- Validation de l'efficacité des algorithmes d'IA pour la sécurité
- Interface utilisateur accessible pour la surveillance en temps réel
- Documentation des performances et limites du système

## Applications

Ce projet démontre l'application de l'intelligence artificielle dans :
- La surveillance de sécurité automatisée
- L'analyse comportementale en temps réel
- La gestion intelligente des espaces publics
- Le traitement vidéo avec apprentissage automatique

---

**Projet académique - Vision par Ordinateur avec partie de Traitement automatique du langage naturel**
