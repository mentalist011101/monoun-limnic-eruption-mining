"""Phase 3 : analyse des règles FP-Growth (Weka) et de la signature pré-éruption.

Chaîne : lecture des règles Weka -> support du conséquent -> filtrage
(règles informatives, non redondantes) -> fenêtre pré-éruption -> signature
-> fréquence de réapparition par an -> test de tendance.

Critères utilisés : uniquement le support et la confiance du cours.
Une règle X -> Y est jugée informative si conf(X -> Y) > supp(Y).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

ITEM_COLUMNS = ["Season", "Rain", "Tmin", "DTR", "Wind", "Pressure"]

_RULE = re.compile(
    r"^\s*\d+\.\s+\[(.*?)\]:\s+(\d+)\s+==>\s+\[(.*?)\]:\s+(\d+)\s+<conf:\(([\d.]+)\)>"
)


def _items(text):
    """'[A=1, B=1]' -> frozenset({'A', 'B'})."""
    return frozenset(x.strip().split("=")[0] for x in text.split(","))


def parse_weka_rules(path):
    """Lit un fichier de résultats FP-Growth de Weka et renvoie un DataFrame de règles.

    Colonnes : premise, consequent (frozensets d'items), n_premise, n_rule, conf.
    """
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            m = _RULE.match(line)
            if m:
                rows.append({
                    "premise": _items(m.group(1)),
                    "consequent": _items(m.group(3)),
                    "n_premise": int(m.group(2)),
                    "n_rule": int(m.group(4)),
                    "conf": float(m.group(5)),
                })
    return pd.DataFrame(rows)


def load_transactions(path):
    """Lit transactions.csv (YEAR, DOY + 6 colonnes d'items) produit en phase 1."""
    return pd.read_csv(path)


def transaction_sets(transactions):
    """Une transaction par jour, sous forme de frozenset d'items."""
    return [frozenset(row) for row in transactions[ITEM_COLUMNS].itertuples(index=False)]


def contains(transactions, itemset):
    """Masque booléen : jours dont la transaction contient tous les items du motif."""
    mask = pd.Series(True, index=transactions.index)
    for item in itemset:
        mask &= (transactions[ITEM_COLUMNS] == item).any(axis=1)
    return mask


def itemset_support(transactions, itemset):
    """supp(M) = n(M) / n, définition 1.3 du cours."""
    return contains(transactions, itemset).mean()


def add_measures(rules, transactions):
    """Ajoute supp(règle), supp(conséquent) et l'écart conf(X->Y) - supp(Y)."""
    rules = rules.copy()
    n = len(transactions)
    cache = {}
    for cons in rules["consequent"].unique():
        cache[cons] = itemset_support(transactions, cons)
    rules["supp"] = rules["n_rule"] / n
    rules["supp_consequent"] = rules["consequent"].map(cache)
    rules["gain"] = rules["conf"] - rules["supp_consequent"]
    return rules


def filter_informative(rules):
    """Garde les règles où conf(X -> Y) > supp(Y) : X rend Y plus fréquent que d'habitude."""
    return rules[rules["gain"] > 0].reset_index(drop=True)


def remove_redundant(rules):
    """Retire une règle X -> Y s'il existe une règle X' -> Y avec X' ⊂ X et une confiance au moins égale.

    Ajouter des items à la prémisse sans gagner en confiance n'apporte rien
    (cas typique : ajouter Tmin_Warm, présent 95 % du temps).
    """
    keep = []
    by_cons = rules.groupby(rules["consequent"].map(lambda s: tuple(sorted(s))))
    for _, group in by_cons:
        recs = list(group[["premise", "conf"]].itertuples())
        for r in recs:
            redundant = any(o.premise < r.premise and o.conf >= r.conf for o in recs)
            if not redundant:
                keep.append(r.Index)
    return rules.loc[sorted(keep)].reset_index(drop=True)


def rules_with_items(rules, items):
    """Règles dont la prémisse ou le conséquent contient au moins un des items donnés."""
    items = set(items)
    mask = rules.apply(lambda r: bool(items & (r.premise | r.consequent)), axis=1)
    return rules[mask].reset_index(drop=True)


def pre_event_window(transactions, year=1984, doy_event=228, n_days=7):
    """Jours précédant l'éruption, jour de l'éruption inclus (DOY doy_event - n_days + 1 à doy_event)."""
    mask = (transactions["YEAR"] == year) & transactions["DOY"].between(doy_event - n_days + 1, doy_event)
    return transactions[mask]


def common_itemset(days):
    """Items présents tous les jours de la fenêtre : la signature observée."""
    sets = transaction_sets(days)
    return frozenset.intersection(*sets)


def rules_verified_by(rules, days, min_share=1.0):
    """Règles vérifiées (prémisse ∪ conséquent inclus dans la transaction) par au moins min_share des jours."""
    sets = transaction_sets(days)
    share = rules.apply(
        lambda r: sum((r.premise | r.consequent) <= s for s in sets) / len(sets), axis=1
    )
    out = rules.assign(share_window=share)
    return out[out["share_window"] >= min_share].reset_index(drop=True)


def yearly_counts(transactions, itemset):
    """Nombre de jours par an où le motif est présent (années sans occurrence = 0)."""
    mask = contains(transactions, itemset)
    years = range(transactions["YEAR"].min(), transactions["YEAR"].max() + 1)
    return transactions[mask].groupby("YEAR").size().reindex(years, fill_value=0).rename("days")


def trend_test(counts):
    """Tendance d'une série annuelle : pente linéaire (jours/an) et test de Mann-Kendall (tau de Kendall)."""
    years = counts.index.to_numpy()
    lin = stats.linregress(years, counts.to_numpy())
    tau = stats.kendalltau(years, counts.to_numpy())
    return pd.Series({
        "slope_days_per_year": lin.slope,
        "slope_days_per_decade": lin.slope * 10,
        "linreg_pvalue": lin.pvalue,
        "kendall_tau": tau.statistic,
        "mann_kendall_pvalue": tau.pvalue,
    })


def plot_yearly(counts, title, path=None, mark_year=1984):
    """Barres du nombre de jours par an, droite de tendance, année de l'éruption en évidence."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#c0392b" if y == mark_year else "#4a6fa5" for y in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    lin = stats.linregress(counts.index, counts.values)
    ax.plot(counts.index, lin.intercept + lin.slope * counts.index, color="#333333",
            linestyle="--", linewidth=1, label=f"tendance : {lin.slope * 10:+.2f} jours/décennie")
    ax.set_title(title)
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre de jours")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    return fig
