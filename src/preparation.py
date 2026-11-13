"""Phase 1 : préparation des données NASA POWER du lac Monoun.

Chaîne : chargement -> contrôle des manquants -> DTR -> discrétisation
-> transactions (6 items par jour) -> binaire (18 colonnes 0/1) -> ARFF pour Weka.

Règle des bornes : une valeur égale à un seuil va dans la classe intermédiaire
(l'énoncé définit les classes extrêmes par des inégalités strictes).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SENTINEL = -999

# Pour chaque variable : (colonne source, seuil bas, seuil haut,
# (item < bas, item dans [bas, haut], item > haut))
THRESHOLDS = {
    "Rain": ("PRECTOTCORR", 1.0, 20.0,
             ("Rain_None", "Rain_Moderate", "Rain_Extreme_Downpour")),
    "Tmin": ("T2M_MIN", 11.0, 14.0,
             ("Tmin_Severe_Cooling", "Tmin_Cold", "Tmin_Warm")),
    "DTR": ("DTR", 8.0, 14.0,
            ("DTR_Suppressed", "DTR_Moderate", "DTR_Extreme")),
    "Wind": ("WS2M", 1.5, 3.0,
             ("Wind_Calm", "Wind_Moderate", "Wind_High_Shear")),
    "Pressure": ("PS", 88.4, 88.7,
                 ("Pressure_Critical_Drop", "Pressure_Normal", "Pressure_High_Stable")),
}

SEASON_ITEMS = ("Season_Dry", "Season_Transition", "Season_Monsoon_Peak")


def load_power_csv(path):
    """Lit un CSV NASA POWER en sautant l'en-tête (-BEGIN HEADER- ... -END HEADER-)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip() == "-END HEADER-":
                header_end = i
                break
        else:
            raise ValueError(f"Ligne '-END HEADER-' introuvable dans {path}")
    return pd.read_csv(path, skiprows=header_end + 1)


def check_missing(df, sentinel=SENTINEL):
    """Remplace le code sentinelle par NaN et renvoie (df, nombre de manquants par colonne)."""
    df = df.replace(sentinel, np.nan)
    return df, df.isna().sum()


def add_dtr(df):
    """Ajoute l'amplitude thermique journalière DTR = T2M_MAX - T2M_MIN (arrondie à 2 décimales)."""
    df = df.copy()
    # Arrondi : évite qu'une différence comme 8.000000001 sorte de la classe intermédiaire.
    df["DTR"] = (df["T2M_MAX"] - df["T2M_MIN"]).round(2)
    return df


def discretize(series, low, high, labels):
    """Découpe une variable en 3 items : < low, [low, high], > high."""
    below, middle, above = labels
    return pd.Series(
        np.select([series < low, series > high], [below, above], default=middle),
        index=series.index,
    )


def discretize_season(doy):
    """Saison selon le jour de l'année (sèche : DOY >= 320 ou <= 60)."""
    dry, transition, monsoon = SEASON_ITEMS
    return pd.Series(
        np.select([(doy >= 320) | (doy <= 60), doy <= 150], [dry, transition], default=monsoon),
        index=doy.index,
    )


def build_transactions(df, thresholds=THRESHOLDS):
    """Renvoie un DataFrame de 6 colonnes (une par variable) : un item par jour et par variable."""
    trans = pd.DataFrame({"Season": discretize_season(df["DOY"])}, index=df.index)
    for name, (col, low, high, labels) in thresholds.items():
        trans[name] = discretize(df[col], low, high, labels)
    return trans


def item_frequencies(transactions):
    """Effectif et proportion de chaque item, variable par variable."""
    freq = (
        transactions.melt(var_name="variable", value_name="item")
        .value_counts()
        .rename("count")
        .reset_index()
    )
    freq["share"] = (freq["count"] / len(transactions)).round(4)
    return freq.sort_values(["variable", "count"], ascending=[True, False], ignore_index=True)


def all_items(thresholds=THRESHOLDS):
    """Liste ordonnée des 18 items possibles."""
    items = list(SEASON_ITEMS)
    for _, _, _, labels in thresholds.values():
        items.extend(labels)
    return items


def to_binary(transactions, thresholds=THRESHOLDS):
    """Encode les transactions en 18 colonnes 0/1 (présence de l'item ce jour-là).

    Tous les items sont conservés, même ceux qui n'apparaissent jamais.
    """
    binary = pd.get_dummies(transactions, prefix="", prefix_sep="").astype(int)
    binary = binary.T.groupby(level=0).sum().T  # fusionne d'éventuelles colonnes en double
    binary = binary.reindex(columns=all_items(thresholds), fill_value=0)
    n_vars = transactions.shape[1]
    if not (binary.sum(axis=1) == n_vars).all():
        raise ValueError(f"Chaque jour doit contenir exactement {n_vars} items.")
    return binary


def export_arff(binary, path, relation="monoun_transactions"):
    """Écrit le tableau binaire en ARFF (attributs {0,1}). Dans Weka FPGrowth : positiveIndex = 2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"@relation {relation}\n\n")
        for col in binary.columns:
            f.write(f"@attribute {col} {{0,1}}\n")
        f.write("\n@data\n")
        for row in binary.itertuples(index=False):
            f.write(",".join(map(str, row)) + "\n")
    return path


def export_arff_nominal(transactions, path, relation="monoun_transactions_nominal",
                        thresholds=THRESHOLDS):
    """Écrit les transactions en ARFF avec 6 attributs nominaux (3 valeurs chacun).

    Format destiné à Apriori dans Weka : chaque valeur nominale est un item
    (ex. Rain=Rain_Extreme_Downpour). Avec le format binaire {0,1}, Apriori
    traiterait aussi les 0 (absences) comme des items.
    """
    values = {"Season": SEASON_ITEMS}
    values.update({name: labels for name, (_, _, _, labels) in thresholds.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"@relation {relation}\n\n")
        for col in transactions.columns:
            f.write(f"@attribute {col} {{{','.join(values[col])}}}\n")
        f.write("\n@data\n")
        for row in transactions.itertuples(index=False):
            f.write(",".join(row) + "\n")
    return path


def save_index(df, path):
    """Sauvegarde la correspondance ligne de transaction -> (YEAR, DOY) pour la phase 3."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["YEAR", "DOY"]].reset_index(drop=True).rename_axis("row_id").to_csv(path)
    return path
