Générateur de Rapports NLP
==========================

.. currentmodule:: nlp_report_generator

.. autoclass:: NLPReportGenerator
   :members:

Aperçu
------

La classe ``NLPReportGenerator`` fournit une génération automatisée de rapports de surveillance avec des capacités de traitement du langage naturel. Elle analyse les événements de sécurité et produit des résumés quotidiens complets avec évaluation des menaces et recommandations opérationnelles.

Démarrage Rapide
----------------

.. code-block:: python

   from nlp_report_generator import NLPReportGenerator
   
   # Initialiser le générateur
   generator = NLPReportGenerator()
   
   # Enregistrer des événements
   generator.log_event("alerte_securite", "Mouvement détecté en zone restreinte")
   generator.log_event("vol", "Incident de vol à l'étalage signalé")
   
   # Générer le rapport
   rapport = generator.generate_daily_summary()
   print(rapport)

Fonctionnalités Principales
---------------------------

* **Classification Automatique des Menaces** : Événements catégorisés en risque Élevé/Moyen/Faible
* **Analyse de Motifs** : Identifie les heures de pointe et tendances d'activité
* **Rapports en Langage Naturel** : Résumés lisibles avec insights actionnables
* **Surveillance des Foules** : Gestion spéciale des événements liés aux foules
* **Recommandations Contextuelles** : Suggestions de sécurité basées sur les motifs d'événements

Référence API
-------------

Méthodes Principales
~~~~~~~~~~~~~~~~~~~~

.. method:: log_event(event_type, details, timestamp=None)

   Enregistre les événements de surveillance pour la génération de rapports.
   
   :param str event_type: Catégorie de l'événement
   :param str details: Description de l'événement
   :param datetime timestamp: Horodatage optionnel (par défaut maintenant)

.. method:: generate_daily_summary(target_date=None)

   Génère un rapport de surveillance quotidien complet.
   
   :param str target_date: Date au format AAAA-MM-JJ (par défaut aujourd'hui)
   :returns: Chaîne de rapport formatée
   :rtype: str

Classification des Menaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Les événements sont automatiquement classifiés en trois niveaux de menace :

* **Priorité Élevée** 🔴 : tir, agression, bagarre, abus, explosion, incendie criminel
* **Priorité Moyenne** 🟡 : vol qualifié, cambriolage, vol, vol à l'étalage, vandalisme
* **Priorité Faible** 🟢 : Tous les autres événements

Structure du Rapport
--------------------

Les rapports générés incluent :

1. **Évaluation des Menaces** - Niveau de risque codé par couleur
2. **Résumé Exécutif** - Vue d'ensemble de la situation
3. **Métriques Opérationnelles** - Statistiques d'activité
4. **Journal des Événements Critiques** - Incidents prioritaires
5. **Recommandations** - Conseils de sécurité actionnables

Exemple de Sortie
-----------------

.. code-block:: text

   📊 RAPPORT DE SURVEILLANCE QUOTIDIEN - 2024-12-15
   ==================================================
   
   🎯 ÉVALUATION DES MENACES : 🟡 RISQUE MOYEN
   📝 RÉSUMÉ : Préoccupations de sécurité modérées avec 2 événements priorité moyenne.
   
   📈 MÉTRIQUES OPÉRATIONNELLES :
      • Événements Totaux : 8
      • Heures de Pointe : 14:00, 18:00
   
   💡 RECOMMANDATIONS :
      • Augmenter la surveillance pendant les heures de pointe
      • Réviser les protocoles de sécurité pour la prévention des vols

Intégration Streamlit
--------------------

.. function:: display_daily_report_section(manager)

   Composant UI Streamlit pour la génération et téléchargement de rapports.
   
   :param manager: Gestionnaire système avec attribut nlp_reporter

Usage dans la barre latérale Streamlit :

.. code-block:: python

   with st.sidebar:
       display_daily_report_section(gestionnaire_surveillance)

Configuration
-------------

Le système de mots-clés de menaces peut être personnalisé :

.. code-block:: python

   generator = NLPReportGenerator()
   
   # Ajouter des mots-clés de menace personnalisés
   generator.threat_keywords['high_threat'].extend(['arme', 'violence'])
   generator.threat_keywords['medium_threat'].extend(['intrusion', 'flânerie'])

Bonnes Pratiques
----------------

1. **Enregistrement Régulier** : Enregistrer les événements immédiatement pour des horodatages précis
2. **Types d'Événements Cohérents** : Utiliser des catégories standardisées
3. **Descriptions Détaillées** : Fournir des détails spécifiques pour une meilleure classification
4. **Génération Quotidienne** : Générer les rapports à intervalles réguliers
5. **Archivage** : Sauvegarder les rapports pour analyse historique

Dépendances
-----------

Packages Python requis :

.. code-block:: text

   datetime
   collections.defaultdict
   collections.Counter
   streamlit  # pour l'intégration UI

Gestion des Erreurs
-------------------

Le système inclut une gestion d'erreurs intégrée :

* **Aucun Événement** : Message approprié si aucun événement enregistré
* **Horodatages Invalides** : Utilise l'heure actuelle par défaut
* **Analyse Vide** : Gère les cas sans événements correspondant aux critères
* **Données Manquantes** : Traitement gracieux des attributs manquants

.. note::
   Ce système est conçu pour les applications de surveillance sécuritaire. 
   Assurez-vous de la conformité avec les réglementations locales de confidentialité.

.. warning::
   Les menaces de haute priorité doivent déclencher une révision manuelle immédiate 
   en plus de la génération automatique de rapports.
