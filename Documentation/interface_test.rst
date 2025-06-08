GuardVision Test - Test du Système de Surveillance Intelligent en Temps Réel
==================================================================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
    :target: https://python.org
.. image:: https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B
    :target: https://streamlit.io
.. image:: https://img.shields.io/badge/Licence-MIT-green
    :target: LICENSE

Check the Test code : `Test Interface <../Code_test/model_managing.py>`_

Un système de surveillance combinant reconnaissance d'actions et analyse de densité de foule utilisant des modèles TensorFlow.

Fonctionnement
--------------

Architecture du Système
~~~~~~~~~~~~~~~~~~~~~~
1. **Entrée Vidéo** : Capture en direct depuis webcam/caméra IP
2. **Traitement par Modèles** :
   - Reconnaissance d'Actions : Détecte les comportements suspects
   - Densité de Foule : Estime le nombre de personnes
3. **Système d'Alerte** : Déclenche des avertissements visuels

.. figure:: images/architecture2.png
    :width: 80%
    :align: center
    :alt: Architecture Système

    Diagramme de flux de données

Composants Clés
~~~~~~~~~~~~~~~
- ``ModelManager`` : Gère le chargement des modèles et l'inférence multi-thread
- ``draw_predictions()`` : Visualise les détections sur le flux vidéo
- Interface Dynamique : Métriques de performance et contrôles en temps réel


Installation
------------
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/med-reda-nk/GuardVision.git
   cd guardvision-pro
