"""Génère les figures, tableaux (.tex) et macros chiffrées du rapport LaTeX.

Tous les nombres du rapport viennent d'ici : rien n'est recopié à la main.
Usage (depuis la racine du dépôt) : python reports/latex/generate_assets.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analyse import (  # noqa: E402
    ITEM_COLUMNS, add_measures, common_itemset, contains, filter_informative, itemset_support,
    load_transactions, parse_weka_rules, pre_event_window, remove_redundant, rules_verified_by,
    rules_with_items, trend_test, yearly_counts,
)
from src.preparation import add_dtr, load_power_csv  # noqa: E402

OUT = Path(__file__).resolve().parent
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

# Palette : bleu (série unique), rouge pour l'année 1984, gris pour le recessif.
BLUE, ORANGE, RED, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#e34948", "#b5b4ad", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})

EXTREMES = ["Rain_Extreme_Downpour", "Pressure_Critical_Drop", "Tmin_Cold", "Tmin_Severe_Cooling",
            "DTR_Extreme", "Wind_High_Shear"]
SEASONS = ["Season_Dry", "Season_Transition", "Season_Monsoon_Peak"]
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


# ---------------------------------------------------------------- utilitaires
def fr(x, nd=2):
    """Nombre au format français (virgule décimale, espace fine pour les milliers)."""
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", "\\,")
    s = f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", "\\,")
    return "$-$" + s[1:] if s.startswith("-") else s


def it(name):
    """Nom d'item en police machine, échappé pour LaTeX."""
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def tt(v):
    return "\\texttt{" + v.replace("_", "\\_") + "}"


def items(s):
    return ", ".join(it(x) for x in sorted(s))


def save_table(name, header, rows, colspec, size="\\small"):
    lines = [size, f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             "\\rowcolor{NuitBleue!90}",
             " & ".join(f"\\color{{white}}\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
        else:
            lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


macros = {}


def macro(name, value):
    macros[name] = value


def savefig(fig, name):
    from matplotlib.ticker import FuncFormatter
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()).__name__ in ("ScalarFormatter", "LogFormatterSciNotation"):
                axis.set_major_formatter(comma)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- données
raw = add_dtr(load_power_csv(ROOT / "data" / "raw" / "LAC_MONOUN_POWER_Daily_1983-2024.csv"))
raw["DATE"] = pd.to_datetime(raw["YEAR"].astype(str), format="%Y") + pd.to_timedelta(raw["DOY"] - 1, unit="D")
raw["MONTH"] = raw["DATE"].dt.month
tx = load_transactions(ROOT / "data" / "processed" / "transactions.csv")
N = len(tx)
macro("Njours", fr(N))
macro("Nmanquants", str(int((raw[["PRECTOTCORR", "T2M_MIN", "T2M_MAX", "WS2M", "PS"]] == -999).sum().sum())))

VARS = {
    "PRECTOTCORR": ("Précipitations", "mm/j"), "T2M_MIN": ("Température minimale", "°C"),
    "T2M_MAX": ("Température maximale", "°C"), "DTR": ("Amplitude thermique", "°C"),
    "WS2M": ("Vitesse du vent", "m/s"), "PS": ("Pression de surface", "kPa"),
}

# ---------------------------------------------------------------- 1. statistiques descriptives
rows = []
for v, (lab, unit) in VARS.items():
    d = raw[v]
    rows.append([tt(v), lab, unit, fr(d.mean()), fr(d.std()), fr(d.min()), fr(d.median()), fr(d.max())])
save_table("tab_descriptives", ["Variable", "Signification", "Unité", "Moy.", "É.-type", "Min", "Méd.", "Max"],
           rows, "lllrrrrr", size="\\footnotesize")

# ---------------------------------------------------------------- 2. climatologie mensuelle
clim = raw.groupby("MONTH")[["PRECTOTCORR", "T2M_MIN", "DTR", "WS2M", "PS"]].mean()
fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.6))
for ax, v in zip(axes, clim.columns):
    lab, unit = VARS[v]
    ax.axvspan(5.5, 11.0, color="#f1f0eb", zorder=0)
    ax.plot(clim.index, clim[v], color=BLUE, linewidth=2, marker="o", markersize=4)
    ax.scatter([8], [clim.loc[8, v]], s=40, color=RED, zorder=5)
    ax.set_title(f"{lab} ({unit})")
    ax.set_xticks(range(1, 13), MONTHS)
    ax.grid(axis="x", visible=False)
fig.text(0.5, -0.06, "Zone grisée : Season_Monsoon_Peak (DOY 151–319, de juin à mi-novembre). Point rouge : août, mois de l'éruption.",
         ha="center", color=MUTED, fontsize=8.5)
fig.tight_layout()
savefig(fig, "fig_climatologie")
macro("PSaout", fr(clim.loc[8, "PS"], 3))
macro("PSjanv", fr(clim.loc[1, "PS"], 3))
macro("PSmax", fr(clim["PS"].max(), 3))
macro("PSmaxMois", ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
                    "octobre", "novembre", "décembre"][int(clim["PS"].idxmax()) - 1])
macro("PSmin", fr(clim["PS"].min(), 3))
macro("PSminMois", ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
                    "octobre", "novembre", "décembre"][int(clim["PS"].idxmin()) - 1])
macro("PluieAout", fr(clim.loc[8, "PRECTOTCORR"], 1))
macro("TminAout", fr(clim.loc[8, "T2M_MIN"], 1))

# ---------------------------------------------------------------- 3. histogrammes et seuils
TH = {"PRECTOTCORR": (1, 20), "T2M_MIN": (11, 14), "DTR": (8, 14), "WS2M": (1.5, 3), "PS": (88.4, 88.7)}
fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.6))
for ax, (v, (a, b)) in zip(axes, TH.items()):
    lab, unit = VARS[v]
    data = raw[v].clip(upper=60) if v == "PRECTOTCORR" else raw[v]
    ax.hist(data, bins=50, color=BLUE, edgecolor="white", linewidth=0.4)
    for t in (a, b):
        ax.axvline(t, color=INK, linestyle="--", linewidth=1)
    ax.set_title(f"{lab} ({unit})")
    ax.set_yticks([])
    ax.grid(visible=False)
    ymax = ax.get_ylim()[1]
    for t in (a, b):
        ax.text(t, ymax * 0.97, f" {str(t).replace('.', ',')}", fontsize=7.5, color=INK, va="top")
fig.text(0.5, -0.06, "Lignes en pointillés : seuils de discrétisation de l'énoncé. Pluie tronquée à 60 mm pour la lisibilité.",
         ha="center", color=MUTED, fontsize=8.5)
fig.tight_layout()
savefig(fig, "fig_histogrammes")

# seuils vs centiles
rows = [
    [it("Rain_Extreme_Downpour"), "> 20 mm", "P95", f"{fr(raw.PRECTOTCORR.quantile(.95), 1)} mm", "Proche"],
    [it("Tmin_Severe_Cooling"), "< 11 °C", "P10", f"{fr(raw.T2M_MIN.quantile(.10), 1)} °C", "\\textcolor{RougeL}{Non}"],
]
save_table("tab_centiles", ["Item", "Seuil énoncé", "Centile annoncé", "Centile réel", "Conforme ?"], rows, "lcccc",
           size="\\footnotesize")
macro("PnfRain", fr(raw.PRECTOTCORR.quantile(.95), 1))
macro("PdixTmin", fr(raw.T2M_MIN.quantile(.10), 1))
macro("PsoixquinzePS", fr(raw.PS.quantile(.75), 2))

# égalités aux seuils
rows = []
for v, (a, b) in TH.items():
    for t in (a, b):
        n = int((raw[v].round(2) == t).sum())
        rows.append([tt(v), fr(t, 1 if t != int(t) else 0).replace("\\,", ""), n])
save_table("tab_egalites", ["Variable", "Seuil", "Jours égaux au seuil"], rows, "lcr")

# ---------------------------------------------------------------- 4. items
share = tx[ITEM_COLUMNS].melt()["value"].value_counts()
order = [("Season", SEASONS),
         ("Rain", ["Rain_None", "Rain_Moderate", "Rain_Extreme_Downpour"]),
         ("Tmin", ["Tmin_Warm", "Tmin_Cold", "Tmin_Severe_Cooling"]),
         ("DTR", ["DTR_Suppressed", "DTR_Moderate", "DTR_Extreme"]),
         ("Wind", ["Wind_Calm", "Wind_Moderate", "Wind_High_Shear"]),
         ("Pressure", ["Pressure_High_Stable", "Pressure_Normal", "Pressure_Critical_Drop"])]
COND = {"Season_Dry": "DOY $\\geq$ 320 ou $\\leq$ 60", "Season_Transition": "61 -- 150",
        "Season_Monsoon_Peak": "151 -- 319", "Rain_None": "< 1 mm", "Rain_Moderate": "[1 ; 20] mm",
        "Rain_Extreme_Downpour": "> 20 mm", "Tmin_Warm": "> 14 °C", "Tmin_Cold": "[11 ; 14] °C",
        "Tmin_Severe_Cooling": "< 11 °C", "DTR_Suppressed": "< 8 °C", "DTR_Moderate": "[8 ; 14] °C",
        "DTR_Extreme": "> 14 °C", "Wind_Calm": "< 1,5 m/s", "Wind_Moderate": "[1,5 ; 3] m/s",
        "Wind_High_Shear": "> 3 m/s", "Pressure_High_Stable": "> 88,7 kPa",
        "Pressure_Normal": "[88,4 ; 88,7] kPa", "Pressure_Critical_Drop": "< 88,4 kPa"}
SRC = {"Season": "DOY", "Rain": "PRECTOTCORR", "Tmin": "T2M\\_MIN", "DTR": "DTR", "Wind": "WS2M",
       "Pressure": "PS"}
rows = []
for var, its in order:
    for i, x in enumerate(its):
        n = int(share.get(x, 0))
        rows.append([f"\\texttt{{{SRC[var]}}}" if i == 0 else "", it(x), COND[x], fr(n), f"{fr(100 * n / N, 1)} \\%"])
    rows.append("\\addlinespace[2pt]")
save_table("tab_items", ["Variable", "Item", "Condition", "Jours", "Part"], rows[:-1], "lllrr")
for x in share.index:
    macro("n" + x.replace("_", ""), fr(int(share[x])))
    macro("p" + x.replace("_", ""), fr(100 * share[x] / N, 1))

# figure : parts des items
fig, ax = plt.subplots(figsize=(8, 5.2))
labels, vals = [], []
for var, its in order[::-1]:
    for x in its[::-1]:
        labels.append(x)
        vals.append(100 * share.get(x, 0) / N)
y = np.arange(len(labels))
colors = [RED if x in EXTREMES else BLUE for x in labels]
ax.barh(y, vals, color=colors, height=0.7)
ax.set_yticks(y, labels, fontsize=8, family="DejaVu Sans Mono")
for yi, v, lab in zip(y, vals, labels):
    txt = f"{v:.1f} %".replace(".", ",") if v >= 0.5 else f"{int(share.get(lab, 0))} jours"
    ax.text(v + 0.8, yi, txt, va="center", fontsize=7.5, color=RED if lab in EXTREMES else INK)
ax.set_xlabel("Part des jours (%)")
ax.set_xlim(0, 105)
ax.grid(axis="y", visible=False)
ax.text(104, len(labels) - 0.5, "rouge : items « extrêmes »", ha="right", va="top", color=RED, fontsize=8)
savefig(fig, "fig_items")

# items extrêmes par saison
rows = []
for x in [e for e in EXTREMES]:
    sub = tx[(tx[ITEM_COLUMNS] == x).any(axis=1)]
    tot = len(sub)
    cs = sub["Season"].value_counts()
    rows.append([it(x), fr(tot)] + [f"{fr(100 * cs.get(s, 0) / tot, 0)} \\%" if tot else "--" for s in SEASONS])
save_table("tab_extremes_saisons", ["Item extrême", "Jours", "Sèche", "Transition", "Mousson"], rows, "lrrrr")

# ---------------------------------------------------------------- 5. règles
runs, summary = {}, []
for s in ["010", "005", "001"]:
    r = add_measures(parse_weka_rules(ROOT / "weka" / "results" / f"fpgrowth_s{s}_c070.txt"), tx)
    inf = filter_informative(r)
    clean = remove_redundant(inf)
    runs[s] = (r, inf, clean)
    n_ext = len(rules_with_items(r, EXTREMES))
    summary.append([fr(int(s) / 100), f"$\\geq$ {fr(int(np.ceil(int(s) / 100 * N)))}", fr(len(r)),
                    fr(len(inf)), fr(len(clean)), fr(n_ext)])
save_table("tab_essais", ["minSupp", "Jours min.", "Règles Weka", "Informatives", "Non redond.", "Avec item extrême"],
           summary, "cccccc", size="\\footnotesize")
rall, rinf, rules = runs["001"]
macro("NreglesBrutes", fr(len(rall)))
macro("NreglesInf", fr(len(rinf)))
macro("NreglesClean", fr(len(rules)))
macro("NreglesNonInf", fr(len(rall) - len(rinf)))
macro("NreglesTminWarm", fr(int((rall["consequent"] == frozenset({"Tmin_Warm"})).sum())))

# exemple de calcul
ex = rall[(rall["premise"] == frozenset({"Tmin_Cold"})) & (rall["consequent"] == frozenset({"Rain_None", "Season_Dry"}))].iloc[0]
macro("exNX", fr(int(ex.n_premise)))
macro("exNXY", fr(int(ex.n_rule)))
macro("exSupp", fr(ex.n_rule / N, 4))
macro("exConf", fr(ex.n_rule / ex.n_premise, 3))
nY = int(contains(tx, ["Rain_None", "Season_Dry"]).sum())
macro("exNY", fr(nY))
macro("exSuppY", fr(nY / N, 3))
macro("exGain", fr(ex.n_rule / ex.n_premise - nY / N, 3))
macro("exLift", fr((ex.n_rule / ex.n_premise) / (nY / N), 2))

# exemple de règle trompeuse (Tmin_Warm en conséquent)
tw = rall[(rall["consequent"] == frozenset({"Tmin_Warm"}))].copy()
low = tw.sort_values("gain").iloc[0]
macro("pieceX", items(low.premise))
macro("pieceConf", fr(low.conf))
macro("pieceSuppY", fr(low.supp_consequent, 3))
macro("pieceGain", fr(low.gain, 3))

# nuage support / confiance
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ni = rall[rall["gain"] <= 0]
ii = rall[rall["gain"] > 0]
ax.scatter(ii["supp"], ii["conf"], s=10, color=BLUE, alpha=0.6, label=f"informatives : conf > supp(Y) ({len(ii)})",
           linewidths=0)
ax.scatter(ni["supp"], ni["conf"], s=22, color=ORANGE, alpha=0.9, label=f"non informatives ({len(ni)})",
           linewidths=0)
ax.set_xscale("log")
ax.set_xlabel("Support de la règle (échelle log)")
ax.set_ylabel("Confiance")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
savefig(fig, "fig_nuage_regles")


def rule_rows(df, k, dedupe=True):
    """Lignes de tableau ; les variantes qui ne diffèrent que par Tmin_Warm sont regroupées."""
    out, seen = [], set()
    for x in df.itertuples():
        key = (x.premise - {"Tmin_Warm"}, x.consequent - {"Tmin_Warm"}) if dedupe else (x.premise, x.consequent)
        if key in seen or not key[1]:
            continue
        seen.add(key)
        out.append([items(x.premise), items(x.consequent), fr(x.supp, 3), fr(x.conf), fr(x.supp_consequent),
                    fr(x.gain, 3)])
        if len(out) == k:
            break
    return out


HDR = ["Prémisse X", "Conséquent Y", "supp", "conf", "supp(Y)", "écart"]
SPEC = ">{\\raggedright\\arraybackslash}p{4.7cm}>{\\raggedright\\arraybackslash}p{4.4cm}rrrr"
for x in ["Rain_Extreme_Downpour", "Pressure_Critical_Drop", "Tmin_Cold", "DTR_Extreme"]:
    df = rules_with_items(rules, [x]).sort_values("gain", ascending=False)
    save_table(f"tab_regles_{x}", HDR, rule_rows(df, 4), SPEC, size="\\footnotesize")
    macro("nr" + x.replace("_", ""), fr(len(df)))

# règles « ordinaires » les plus marquées (sans item extrême)
ordinary = rules[~rules.index.isin(rules_with_items(rules, EXTREMES).index)]
ordinary = rules.loc[[i for i in rules.index
                      if not (set(EXTREMES) & (rules.loc[i, "premise"] | rules.loc[i, "consequent"]))]]
save_table("tab_regles_ordinaires", HDR, rule_rows(ordinary.sort_values("gain", ascending=False), 6), SPEC,
           size="\\footnotesize")

# composites
comp = [("Chute de pression + pluie extrême", ["Pressure_Critical_Drop", "Rain_Extreme_Downpour"]),
        ("Chute de pression + nuit froide", ["Pressure_Critical_Drop", "Tmin_Cold"]),
        ("Pluie extrême + nuit froide", ["Rain_Extreme_Downpour", "Tmin_Cold"]),
        ("Pluie extrême + vent fort", ["Rain_Extreme_Downpour", "Wind_High_Shear"]),
        ("Chute de pression + pluie extrême + nuit froide",
         ["Pressure_Critical_Drop", "Rain_Extreme_Downpour", "Tmin_Cold"])]
rows = []
for lab, its in comp:
    n = int(contains(tx, its).sum())
    rows.append([lab, fr(n), fr(100 * n / N, 2) + " \\%"])
save_table("tab_composites", ["Combinaison", "Jours (1983--2024)", "Support"], rows, "lrr")

# ---------------------------------------------------------------- 6. fenêtre et signature
window = pre_event_window(tx, 1984, 228, 7)
signature = common_itemset(window)
macro("signature", items(signature))
macro("sigSupp", fr(itemset_support(tx, signature), 3))
wraw = raw[(raw["YEAR"] == 1984) & raw["DOY"].between(222, 228)]
rows = []
for (_, rw), (_, tr) in zip(wraw.iterrows(), window.iterrows()):
    d = rw["DATE"].strftime("%d/%m")
    rows.append([d, fr(rw["PRECTOTCORR"]), fr(rw["T2M_MIN"]), fr(rw["DTR"]), fr(rw["WS2M"]), fr(rw["PS"]),
                 f"{it(tr['DTR'])}, {it(tr['Wind'])}"])
save_table("tab_fenetre", ["Date", "Pluie", "Tmin", "DTR", "Vent", "PS", "Items variables"], rows,
           "lrrrrrl", size="\\footnotesize")
verified = rules_verified_by(rules, window).sort_values("gain", ascending=False)
save_table("tab_regles_fenetre", HDR, rule_rows(verified, 10, dedupe=False), SPEC, size="\\footnotesize")

# sensibilité à la taille de la fenêtre
rows = []
for n in [3, 7, 14, 30]:
    w = pre_event_window(tx, 1984, 228, n)
    s = common_itemset(w)
    nex = int(w[ITEM_COLUMNS].isin(EXTREMES).sum().sum())
    c = yearly_counts(tx, s)
    t = trend_test(c)
    rank = int((c > c.loc[1984]).sum()) + 1
    ties = int((c == c.loc[1984]).sum())
    rank_s = f"{rank}/42" + ("*" if ties > 1 else "")
    slope = t.slope_days_per_decade
    rows.append([f"{n} jours", items(s), nex, fr(int(c.loc[1984])), rank_s,
                 fr(0.0 if abs(slope) < 0.05 else slope, 1),
                 fr(t.mann_kendall_pvalue, 3) if t.mann_kendall_pvalue >= 0.001 else "$<$ 0,001"])
save_table("tab_sensibilite", ["Fenêtre", "Signature commune", "Extr.", "1984", "Rang",
                               "Pente", "p (MK)"], rows, "l>{\\raggedright\\arraybackslash}p{4.6cm}ccccc", size="\\footnotesize")

# ---------------------------------------------------------------- 7. réapparition
sig_counts = yearly_counts(tx, signature)
t = trend_test(sig_counts)
macro("sigMoy", fr(sig_counts.mean(), 0))
macro("sigPente", fr(t.slope_days_per_decade, 1))
macro("sigPenteAbs", fr(-t.slope_days_per_decade, 0))
macro("sigTau", fr(t.kendall_tau))
macro("sigAvant", fr(sig_counts.loc[:2013].mean(), 0))
macro("sigApres", fr(sig_counts.loc[2014:].mean(), 0))
macro("sigMilleNeuf", fr(int(sig_counts.loc[1984])))
macro("sigRang", str(int((sig_counts > sig_counts.loc[1984]).sum()) + 1))

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(sig_counts.index, sig_counts.values, color=[RED if y == 1984 else BLUE for y in sig_counts.index],
       width=0.75)
xs = sig_counts.index.to_numpy()
ax.plot(xs, sig_counts.mean() + t.slope_days_per_year * (xs - xs.mean()), color=INK, linestyle="--", linewidth=1.2)
ax.annotate("1984", (1984, sig_counts.loc[1984]), xytext=(1984, sig_counts.loc[1984] + 8), ha="center",
            color=RED, fontsize=8, fontweight="bold")
ax.text(2024.5, sig_counts.max() * 1.12,
        f"- - -  tendance : {t.slope_days_per_decade:+.1f} jours/décennie".replace(".", ","), ha="right",
        fontsize=8, color=INK)
ax.set_ylabel("Jours par an")
ax.set_ylim(0, sig_counts.max() * 1.18)
ax.grid(axis="x", visible=False)
savefig(fig, "fig_signature_par_an")

# items extrêmes par an (petits multiples)
fig, axes = plt.subplots(2, 2, figsize=(10, 4.6), sharex=True)
rows = []
for ax, x in zip(axes.flat, ["Rain_Extreme_Downpour", "Pressure_Critical_Drop", "Tmin_Cold", "DTR_Extreme"]):
    c = yearly_counts(tx, [x])
    tt = trend_test(c)
    ax.bar(c.index, c.values, color=BLUE, width=0.75)
    ax.plot(c.index, c.mean() + tt.slope_days_per_year * (c.index - c.index.to_numpy().mean()), color=INK,
            linestyle="--", linewidth=1)
    p = tt.mann_kendall_pvalue
    title = f"{x}   ({tt.slope_days_per_decade:+.1f} j/déc. ; p = {p:.3f})".replace(".", ",").replace("déc,", "déc.")
    ax.set_title(title, fontsize=9,
                 family="DejaVu Sans Mono", fontweight="normal")
    ax.grid(axis="x", visible=False)
    sig = "oui" if p < 0.05 else "non"
    rows.append([it(x), fr(int(c.sum())), fr(int(c.loc[1984])), fr(tt.slope_days_per_decade, 1),
                 fr(p, 3), sig])
    macro("pente" + x.replace("_", ""), fr(tt.slope_days_per_decade, 1))
    macro("pval" + x.replace("_", ""), fr(p, 3))
for ax in axes[:, 0]:
    ax.set_ylabel("Jours par an")
fig.tight_layout()
savefig(fig, "fig_extremes_par_an")
save_table("tab_tendances", ["Item", "Jours (total)", "Jours 1984", "Pente (j/déc.)", "p (MK)", "Signif. 5 \\%"],
           rows, "lccccc", size="\\footnotesize")

# pression de mousson
ps_mon = raw[raw["DOY"].between(151, 319)].groupby("YEAR")["PS"].mean()
mon = tx[tx["Season"] == "Season_Monsoon_Peak"]
rex = mon[mon["Rain"] == "Rain_Extreme_Downpour"].groupby("YEAR").size().reindex(ps_mon.index, fill_value=0)
macro("psAvant", fr(ps_mon.loc[:2013].mean(), 3))
macro("psApres", fr(ps_mon.loc[2014:].mean(), 3))
macro("psDelta", fr(ps_mon.loc[2014:].mean() - ps_mon.loc[:2013].mean(), 2))
macro("rexAvant", fr(rex.loc[:2013].mean(), 1))
macro("rexApres", fr(rex.loc[2014:].mean(), 1))
fig, ax = plt.subplots(figsize=(9, 3.0))
ax.plot(ps_mon.index, ps_mon.values, color=BLUE, linewidth=2, marker="o", markersize=3.5)
ax.axhline(88.7, color=INK, linestyle="--", linewidth=1)
ax.text(1983, 88.703, "seuil 88,7 kPa (Normal / High_Stable)", fontsize=8, color=INK, va="bottom")
ax.set_ylabel("PS moyenne (kPa)")
ax.grid(axis="x", visible=False)
savefig(fig, "fig_pression_mousson")

tmin_80 = raw[raw["YEAR"] <= 1989]["T2M_MIN"].mean()
tmin_20 = raw[raw["YEAR"] >= 2020]["T2M_MIN"].mean()
macro("tminQuatreVingt", fr(tmin_80, 1))
macro("tminVingt", fr(tmin_20, 1))

# ---------------------------------------------------------------- macros
lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
(OUT / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(list(FIG.glob('*.png')))} figures, {len(list(TAB.glob('*.tex')))} tableaux, {len(macros)} macros")
