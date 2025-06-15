===========================
GuardVision - Guide d'Utilisation
===========================


Description
===========

GuardVision est un système de surveillance intelligent utilisant l'intelligence artificielle pour détecter automatiquement les actions suspectes et analyser la densité de foule en temps réel. Le système combine deux modèles d'IA pour fournir une surveillance complète et génère des rapports détaillés.

Fonctionnalités Principales
============================

* **Reconnaissance d'Actions** : Détection automatique de 13 types d'actions (bagarre, vol, vandalisme, etc.)
* **Analyse de Foule** : Estimation de la densité de foule en temps réel
* **Détection d'arme** : detection de possession d'arme
* **Détection de Poses** : Suivi des poses humaines avec MediaPipe
* **Rapports Automatiques** : Génération de rapports journaliers en langage naturel
* **Interface Web Interactive** : Interface utilisateur moderne avec Streamlit

Pré-requis Système
==================

Configuration Minimale
-----------------------

* **Processeur** : Intel i5 ou équivalent AMD
* **RAM** : 8 GB minimum (16 GB recommandé)
* **GPU** : NVIDIA GTX 1060 ou supérieur (optionnel mais recommandé)
* **Caméra** : Webcam USB ou caméra intégrée
* **Stockage** : 2 GB d'espace libre

Systèmes d'Exploitation Supportés
----------------------------------

* Windows 10/11
* macOS 10.15+
* Ubuntu 18.04+

Installation
============

1. Prérequis Python
--------------------

Vérifiez que Python 3.8+ est installé ::

    python --version

2. Cloner le Projet
--------------------

::

    git clone https://github.com/votre-repo/guardvision.git
    cd guardvision

3. Créer un Environnement Virtuel
----------------------------------

Windows ::

    python -m venv venv
    venv\Scripts\activate

macOS/Linux ::

    python3 -m venv venv
    source venv/bin/activate

4. Installer les Dépendances
-----------------------------

Dépendances Principales ::

    pip install streamlit==1.28.1
    pip install opencv-python==4.8.1.78
    pip install tensorflow==2.13.0
    pip install torch==2.0.1
    pip install torchvision==0.15.2
    pip install mediapipe==0.10.3
    pip install numpy==1.24.3

Dépendances Additionnelles ::

    pip install ultralytics==8.0.165
    pip install Pillow==10.0.0
    pip install pandas==2.0.3

Installation Complète en Une Commande ::

    pip install streamlit==1.28.1 opencv-python==4.8.1.78 tensorflow==2.13.0 torch==2.0.1 torchvision==0.15.2 mediapipe==0.10.3 numpy==1.24.3 ultralytics==8.0.165 Pillow==10.0.0 pandas==2.0.3

5. Structure des Dossiers
--------------------------

Créer la structure suivante ::

    guardvision/
    ├── models/
    │   ├── action_model.keras
    │   └── crowd_model.keras
    ├── streamlit_interface.py (fichier principal)
    ├── requirements.txt
    └── README.md

Configuration des Modèles
=========================

Modèles Requis : téléchargé depuis le lien
--------------

1. **action_model.keras** : Modèle de reconnaissance d'actions
   
   * Format : TensorFlow/Keras (.keras)
   * Classes supportées : 13 actions (Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Normal, Robbery, Shooting, Shoplifting, Stealing, Vandalism)

2. **crowd_model.keras** : Modèle d'analyse de foule
   
   * Format : TensorFlow/Keras (.keras)
   * Output : Densité de foule (valeur numérique ou carte de densité)

Placement des Modèles
---------------------

::

    mkdir models
    # Copier vos modèles dans le dossier models/
    cp votre_modele_action.keras models/action_model.keras
    cp votre_modele_foule.keras models/crowd_model.keras

Utilisation
===========

1. Lancement de l'Application
------------------------------

::

    streamlit run streamlit_interface.py

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse ``http://localhost:8501``

2. Interface Utilisateur
-------------------------

Panneau de Contrôle (Sidebar)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Sélection des Modèles** : Activez/désactivez les modèles individuellement
* **Seuils de Confiance** : Ajustez la sensibilité de détection
* **Paramètres de Performance** : Configurez le saut d'images pour optimiser les performances
* **Résolution Caméra** : Choisissez la résolution d'entrée

Zone Principale
~~~~~~~~~~~~~~~

* **Flux Vidéo en Direct** : Affichage temps réel avec annotations
* **Résultats des Modèles** : Détections actuelles avec scores de confiance
* **Journal des Détections** : Historique des événements détectés

3. Configuration des Seuils
----------------------------

Seuil de Confiance d'Action (0.1 - 1.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **0.5-0.6** : Sensibilité élevée (plus de faux positifs)
* **0.7** : Équilibre recommandé
* **0.8-0.9** : Sensibilité faible (moins de faux positifs)

Seuil de Densité de Foule (0.1 - 2.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **0.3-0.5** : Détection de petits groupes
* **0.5-1.0** : Foules moyennes
* **1.0+** : Foules denses uniquement

4. Paramètres de Performance
-----------------------------

Saut d'Images (Frame Skip)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **1-2** : Traitement de toutes les images (plus précis, plus lent)
* **3-5** : Équilibre performance/précision
* **6-10** : Performance maximale (moins précis)

Fonctionnalités Avancées
=========================

1. Génération de Rapports
--------------------------

Rapport Journalier Automatique
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Le système génère automatiquement des rapports détaillés incluant :

* Résumé exécutif avec niveau de menace
* Métriques opérationnelles
* Analyse des patterns d'activité
* Recommandations contextuelles

Accès aux Rapports
~~~~~~~~~~~~~~~~~~

1. Cliquez sur "Generate Report" dans le panneau de contrôle
2. Visualisez le rapport dans l'expandeur "View Report"
3. Téléchargez avec "Save Report"

2. Types de Détections
-----------------------

Reconnaissance d'Actions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Actions Détectées
   :widths: 25 50 25
   :header-rows: 1

   * - Action
     - Description
     - Niveau de Menace
   * - Normal
     - Activité normale
     - 🟢 Bas
   * - Fighting
     - Combat/bagarre
     - 🔴 Élevé
   * - Assault
     - Agression
     - 🔴 Élevé
   * - Abuse
     - Abus/maltraitance
     - 🔴 Élevé
   * - Shooting
     - Tir d'arme
     - 🔴 Critique
   * - Robbery
     - Vol à main armée
     - 🟡 Moyen
   * - Burglary
     - Cambriolage
     - 🟡 Moyen
   * - Stealing
     - Vol simple
     - 🟡 Moyen
   * - Shoplifting
     - Vol à l'étalage
     - 🟡 Moyen
   * - Vandalism
     - Vandalisme
     - 🟡 Moyen
   * - Arson
     - Incendie criminel
     - 🔴 Élevé
   * - Explosion
     - Explosion
     - 🔴 Critique
   * - Arrest
     - Arrestation
     - 🟡 Moyen

Analyse de Foule
~~~~~~~~~~~~~~~~

* **Densité Faible (0.1-0.5)** : Peu de personnes
* **Densité Moyenne (0.5-1.0)** : Groupe modéré
* **Densité Élevée (1.0-2.0)** : Foule importante
* **Densité Très Élevée (2.0+)** : Surpeuplement critique

Dépannage
=========

Problèmes Communs
-----------------

1. Erreur de Chargement des Modèles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Erreur ::

    Error: Model not found at: models/action_model.keras

**Solution** : Vérifiez que les fichiers de modèles sont dans le bon dossier avec les bons noms.

2. Erreur de Caméra
~~~~~~~~~~~~~~~~~~~

Erreur ::

    Could not open camera. Please check if camera is available.

**Solutions** :

* Vérifiez que la caméra n'est pas utilisée par une autre application
* Testez avec différents indices de caméra (0, 1, 2...)
* Vérifiez les permissions de la caméra

3. Performance Lente
~~~~~~~~~~~~~~~~~~~~

**Solutions** :

* Augmentez le saut d'images (Frame Skip)
* Réduisez la résolution de la caméra
* Fermez les autres applications gourmandes en ressources
* Utilisez un GPU si disponible

4. Erreurs de Dépendances
~~~~~~~~~~~~~~~~~~~~~~~~~

Erreur ::

    ModuleNotFoundError: No module named 'cv2'

**Solution** : Réinstallez les dépendances ::

    pip install --upgrade -r requirements.txt

Optimisation des Performances
------------------------------

Configuration GPU (NVIDIA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Installez CUDA Toolkit 11.8
2. Installez cuDNN 8.6
3. Installez TensorFlow-GPU ::

    pip install tensorflow-gpu==2.13.0

Configuration Mémoire
~~~~~~~~~~~~~~~~~~~~~~

Pour les systèmes avec RAM limitée, ajoutez dans le code avant l'initialisation des modèles ::

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

Versions des Bibliothèques
===========================

Dépendances Principales
-----------------------

.. code-block:: text

    streamlit==1.28.1
    opencv-python==4.8.1.78
    tensorflow==2.13.0
    torch==2.0.1
    torchvision==0.15.2
    mediapipe==0.10.3
    numpy==1.24.3
    ultralytics==8.0.165

Dépendances Utilitaires
-----------------------

.. code-block:: text

    Pillow==10.0.0
    pandas==2.0.3

Bibliothèques Intégrées Python
-------------------------------

* ``queue`` (built-in)
* ``threading`` (built-in)
* ``collections`` (built-in)
* ``datetime`` (built-in)
* ``json`` (built-in)
* ``re`` (built-in)
* ``os`` (built-in)
* ``time`` (built-in)

Fichier requirements.txt
-------------------------

Créez un fichier ``requirements.txt`` avec le contenu suivant ::

    streamlit==1.28.1
    opencv-python==4.8.1.78
    tensorflow==2.13.0
    torch==2.0.1
    torchvision==0.15.2
    mediapipe==0.10.3
    numpy==1.24.3
    ultralytics==8.0.165
    Pillow==10.0.0
    pandas==2.0.3

Puis installez avec ::

    pip install -r requirements.txt

Journal des Modifications
=========================

Version 1.0.0
--------------

* Système dual-modèle (Action + Foule)
* Interface Streamlit complète
* Génération de rapports NLP
* Détection de poses MediaPipe
* Support YOLO pour détection de personnes

