Reconnaissance d'Actions
=========================

Cette composante vise à identifier des comportements suspects ou anormaux pouvant indiquer une menace potentielle.

Méthodologie
------------

- **Modèles utilisés** : Utilisation de DenseNet121 pré-entraîné sur ImageNet comme backbone, en excluant les couches fully-connected supérieures pour l'extraction de caractéristiques, avec un classifieur profond pour classifier les scènes.
- **Classification d'actions** : Identification de comportements spécifiques comme courir de manière erratique, se battre, tomber, ou adopter une posture menaçante.
- **Détection d'anomalies** : Apprentissage de modèles de comportement normal pour identifier les déviations significatives.
- **Analyse de posture** : Utilisation de frameworks comme OpenPose ou MediaPipe pour l'extraction de squelettes et l'analyse de posture.

Actions surveillées
-------------------

- Interactions violentes entre individus
- Mouvements brusques ou erratiques
- Comportements suspects (rôder, surveillance prolongée)
- Chutes ou situations de détresse
- Gestes associés à l'utilisation d'armes

Implémentation
--------------

- Extraction de caractéristiques spatio-temporelles
- Fenêtres glissantes pour l'analyse continue du flux vidéo
- Système de score pour évaluer le niveau de risque associé aux actions détectées
- Fusion des informations de multiples caméras pour une analyse cohérente

Model Performance Analysis
==========================

Overview
--------

This analysis presents the Receiver Operating Characteristic (ROC) curves for a multi-class classification model designed to predict different types of criminal activities. The model demonstrates varying levels of performance across different crime categories, as measured by the Area Under the Curve (AUC) metric.

Model Performance by Crime Category
-----------------------------------

The classification model shows significant variation in predictive performance across different crime types:

**High-Performing Categories (AUC > 0.70):**

* **Shoplifting (AUC: 0.86)** - Exhibits the strongest predictive performance, with the ROC curve showing excellent separation from the random baseline
* **Stealing (AUC: 0.73)** - Demonstrates good predictive capability with substantial improvement over random classification
* **Robbery (AUC: 0.70)** - Shows solid performance with meaningful predictive value

**Moderate-Performing Categories (AUC: 0.60-0.69):**

* **Arson (AUC: 0.67)** - Displays moderate predictive performance with room for improvement
* **Abuse (AUC: 0.64)** - Shows fair classification ability above random chance
* **Normal (AUC: 0.60)** - Represents baseline behavior classification with modest predictive power
* **Arrest (AUC: 0.59)** - Demonstrates limited but measurable predictive capability

**Lower-Performing Categories (AUC < 0.60):**

* **Explosion (AUC: 0.56)** - Shows minimal improvement over random classification
* **Vandalism (AUC: 0.56)** - Exhibits limited predictive performance
* **Assault (AUC: 0.54)** - Demonstrates poor classification ability
* **Burglary (AUC: 0.52)** - Shows marginal performance barely above random chance
* **Shooting (AUC: 0.46)** - Performs below random baseline, indicating potential model issues
* **Fighting (AUC: 0.40)** - Exhibits poor predictive performance significantly below random chance

ROC Curve Interpretation
-------------------------

The ROC curves plot the True Positive Rate (sensitivity) against the False Positive Rate (1-specificity) at various classification thresholds. Key observations include:

**Curve Characteristics:**

* Curves closer to the top-left corner indicate better performance
* The dashed diagonal line represents random guessing (AUC = 0.50)
* Steeper initial rises indicate better performance at low false positive rates
* The area under each curve quantifies overall classification performance

**Performance Distribution:**

The model shows a clear hierarchy of predictive capability, with property crimes (shoplifting, stealing) showing superior performance compared to violent crimes (shooting, fighting, assault). This suggests that the feature set may be more discriminative for certain types of criminal behavior.

Model Implications
------------------

**Strengths:**

* Excellent performance in detecting shoplifting and stealing activities
* Reasonable performance across several crime categories
* Clear differentiation between high and low-performing categories

**Areas for Improvement:**

* Poor performance on violent crimes (shooting, fighting, assault)
* Several categories performing at or below random chance
* Potential need for feature engineering or class-specific modeling approaches

**Recommendations:**

1. **Feature Analysis**: Investigate which features contribute most to high-performing categories
2. **Class Imbalance**: Examine potential class imbalance issues for poor-performing categories
3. **Model Refinement**: Consider ensemble methods or specialized models for underperforming classes
4. **Data Quality**: Review data quality and labeling accuracy for categories with AUC < 0.50

Conclusion
----------

The multi-class classification model demonstrates variable performance across different crime categories, with particularly strong results for property crimes and weaker performance for violent crimes. The significant variation in AUC scores suggests that different crime types may require specialized modeling approaches or additional feature engineering to achieve optimal predictive performance.

Technical Specifications
-------------------------

:Model Type: Multi-class Classification
:Evaluation Metric: Area Under the ROC Curve (AUC)
:Number of Classes: 13 crime categories + 1 normal class
:Performance Range: AUC 0.40 - 0.86
:Baseline Comparison: Random guessing (AUC = 0.50)
