# Lac Monoun : extraction de règles d'association hydro-météorologiques

Extraction par règles d'association (FP-Growth) des combinaisons hydro-météorologiques extrêmes
déclencheuses d'inversion convective au lac Monoun : analyse rétrospective de la catastrophe
du 15 août 1984 (DOY 228) à nos jours.

Projet du cours de Data Mining, M2 Data Science, ENSPY.

## Objectif

1. Transformer 42 ans de données météo journalières (1983–2024) en transactions d'items discrétisés.
2. Extraire avec FP-Growth (Weka) les règles d'association entre conditions météo.
3. Identifier la « signature de risque » ayant précédé l'éruption limnique de 1984 et mesurer
   sa fréquence de réapparition sur 1983–2024, dans le contexte du réchauffement climatique.

## Données

Source : NASA POWER (réanalyse MERRA-2), point journalier, lat 5.58 / lon 10.586, du 01/01/1983 au 31/12/2024.

| Variable      | Description                                   | Unité  |
|---------------|-----------------------------------------------|--------|
| `YEAR`, `DOY` | Année et jour de l'année                      | —      |
| `PRECTOTCORR` | Précipitations corrigées                      | mm/jour|
| `T2M_MIN`     | Température minimale à 2 m                    | °C     |
| `T2M_MAX`     | Température maximale à 2 m                    | °C     |
| `WS2M`        | Vitesse du vent à 2 m                         | m/s    |
| `PS`          | Pression de surface                           | kPa    |

## Démarche

| Partie | Où | Contenu |
|--------|----|---------|
| 1. Préparation | `notebooks/` + `src/` | Nettoyage, calcul de DTR, discrétisation, conversion binaire, export ARFF |
| 2. Extraction  | `weka/`               | FP-Growth dans Weka, réglages et règles obtenues |
| 3. Analyse     | `notebooks/` + `src/` | Signature pré-éruption, fréquence de réapparition, tendance, interprétation |

## Structure

```
.
├── data/
│   ├── raw/          # données brutes NASA POWER (non modifiées)
│   └── processed/    # données nettoyées, transactions, fichiers ARFF
├── notebooks/        # notebooks d'exécution, dans l'ordre des parties
├── src/              # fonctions Python réutilisables, appelées depuis les notebooks
├── weka/
│   └── results/      # règles extraites et paramètres utilisés dans Weka
├── results/
│   └── figures/      # graphiques
└── reports/          # rapport final (PDF) et sources LaTeX (reports/latex/)
```

## Installation

```bash
pip install -r requirements.txt
jupyter notebook
```
