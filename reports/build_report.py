"""Génère le rapport PDF du projet à partir des données et résultats du dépôt.

Usage (depuis la racine du dépôt) : python reports/build_report.py
"""

import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analyse import (  # noqa: E402
    add_measures, common_itemset, filter_informative, itemset_support, load_transactions,
    parse_weka_rules, pre_event_window, remove_redundant, rules_verified_by, rules_with_items,
    trend_test, yearly_counts,
)
from src.preparation import add_dtr, load_power_csv  # noqa: E402

OUT = ROOT / "reports" / "rapport_monoun_fpgrowth.pdf"
FIG = ROOT / "results" / "figures"

# ---------------------------------------------------------------- polices et styles
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
for name, file in [("Serif", "DejaVuSerif.ttf"), ("Serif-Bold", "DejaVuSerif-Bold.ttf"),
                   ("Serif-Italic", "DejaVuSerif-Italic.ttf"), ("Serif-BoldItalic", "DejaVuSerif-BoldItalic.ttf"),
                   ("Sans", "DejaVuSans.ttf"), ("Sans-Bold", "DejaVuSans-Bold.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-Bold",
                              italic="Serif-Italic", boldItalic="Serif-BoldItalic")

ACCENT = colors.HexColor("#1f3b5c")
LIGHT = colors.HexColor("#e8eef5")

S = {
    "body": ParagraphStyle("body", fontName="Serif", fontSize=10, leading=14.5, alignment=TA_JUSTIFY,
                           spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=5, keepWithNext=1),
    "bullet": ParagraphStyle("bullet", fontName="Serif", fontSize=10, leading=14, leftIndent=14,
                             bulletIndent=4, spaceAfter=3, alignment=TA_JUSTIFY),
    "caption": ParagraphStyle("caption", fontName="Serif-Italic", fontSize=8.8, leading=11.5,
                              alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10, textColor=colors.white),
    "formula": ParagraphStyle("formula", fontName="Serif", fontSize=10, leading=15, alignment=TA_CENTER,
                              spaceBefore=2, spaceAfter=6),
    "ref": ParagraphStyle("ref", fontName="Serif", fontSize=9, leading=12.5, leftIndent=14,
                          firstLineIndent=-14, spaceAfter=4),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8, leading=11, backColor=LIGHT,
                           borderPadding=5, spaceBefore=4, spaceAfter=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, align_right=()):
    data = [[Paragraph(str(c), S["cellb" if header and i == 0 else "cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="CENTER")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fb")]),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT))
    t.setStyle(TableStyle(style))
    return t


def figure(path, caption, width=16 * cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), P(caption, "caption")])


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def items_str(s):
    return ", ".join(sorted(s))


# ---------------------------------------------------------------- données et calculs
raw = load_power_csv(ROOT / "data" / "raw" / "LAC_MONOUN_POWER_Daily_1983-2024.csv")
raw = add_dtr(raw)
tx = load_transactions(ROOT / "data" / "processed" / "transactions.csv")
N = len(tx)

runs, summary = {}, []
for s in ["010", "005", "001"]:
    r = add_measures(parse_weka_rules(ROOT / "weka" / "results" / f"fpgrowth_s{s}_c070.txt"), tx)
    inf = filter_informative(r)
    clean = remove_redundant(inf)
    runs[s] = clean
    summary.append([fmt(int(s) / 100), f"≈ {round(int(s) / 100 * N)}", len(r), len(inf), len(clean)])
rules = runs["001"]

window = pre_event_window(tx, 1984, 228, 7)
signature = common_itemset(window)
sig_supp = itemset_support(tx, signature)
verified = rules_verified_by(rules, window).sort_values("gain", ascending=False)
sig_counts = yearly_counts(tx, signature)
sig_trend = trend_test(sig_counts)

EXT = ["Rain_Extreme_Downpour", "Pressure_Critical_Drop", "Tmin_Cold", "DTR_Extreme"]
ext_trends = []
for it in EXT:
    c = yearly_counts(tx, [it])
    t = trend_test(c)
    ext_trends.append([it, int(c.sum()), int(c.loc[1984]), fmt(t.slope_days_per_decade, 1),
                       fmt(t.mann_kendall_pvalue, 3)])

composites = [
    ("Pressure_Critical_Drop + Rain_Extreme_Downpour", ["Pressure_Critical_Drop", "Rain_Extreme_Downpour"]),
    ("Pressure_Critical_Drop + Tmin_Cold", ["Pressure_Critical_Drop", "Tmin_Cold"]),
    ("Rain_Extreme_Downpour + Tmin_Cold", ["Rain_Extreme_Downpour", "Tmin_Cold"]),
    ("Les trois ensemble", ["Pressure_Critical_Drop", "Rain_Extreme_Downpour", "Tmin_Cold"]),
]
comp_rows = [[k, round(itemset_support(tx, v) * N)] for k, v in composites]

ps_mon = raw[raw["DOY"].between(151, 319)].groupby("YEAR")["PS"].mean()
mon = tx[tx["Season"] == "Season_Monsoon_Peak"]
rex = mon[mon["Rain"] == "Rain_Extreme_Downpour"].groupby("YEAR").size().reindex(ps_mon.index, fill_value=0)
ps_before, ps_after = ps_mon.loc[:2013].mean(), ps_mon.loc[2014:].mean()
rex_before, rex_after = rex.loc[:2013].mean(), rex.loc[2014:].mean()
sig_before, sig_after = sig_counts.loc[:2013].mean(), sig_counts.loc[2014:].mean()
tmin_80s = raw[raw["YEAR"] <= 1989]["T2M_MIN"].mean()
tmin_20s = raw[raw["YEAR"] >= 2020]["T2M_MIN"].mean()
p10_tmin = raw["T2M_MIN"].quantile(0.10)
p95_rain = raw["PRECTOTCORR"].quantile(0.95)

item_share = tx.iloc[:, 2:].melt()["value"].value_counts()


def share(it):
    n = int(item_share.get(it, 0))
    return f"{n} ({fmt(100 * n / N, 1)} %)" if n >= 100 else f"{n} jours"


def top_rules(item, k=3):
    r = rules_with_items(rules, [item]).sort_values("gain", ascending=False).head(k)
    return [[items_str(x.premise), items_str(x.consequent), fmt(x.supp, 3), fmt(x.conf),
             fmt(x.supp_consequent), fmt(x.gain)] for x in r.itertuples()]


# ---------------------------------------------------------------- contenu
story = []

# Page de garde
story += [
    Spacer(1, 3 * cm),
    P("École Nationale Supérieure Polytechnique de Yaoundé<br/>Master 2 Data Science", "formula"),
    Spacer(1, 0.4 * cm),
    P("Fouille de données (Data Mining) — Projet 1", "formula"),
    Spacer(1, 2 * cm),
    Paragraph("Extraction par règles d'association des combinaisons hydro-météorologiques extrêmes "
              "déclencheuses d'inversion convective au lac Monoun",
              ParagraphStyle("t", fontName="Sans-Bold", fontSize=18, leading=24, alignment=TA_CENTER,
                             textColor=ACCENT)),
    Spacer(1, 0.6 * cm),
    Paragraph("Analyse rétrospective de la catastrophe du 15 août 1984 à nos jours (1983–2024)",
              ParagraphStyle("st", fontName="Serif-Italic", fontSize=12, leading=16, alignment=TA_CENTER)),
    Spacer(1, 3 * cm),
    P("<b>FOKOUO SAADIE Luciano</b>", "formula"),
    P(f"Année académique 2026–2027 — {date.today().strftime('%d/%m/%Y')}", "formula"),
    Spacer(1, 2.5 * cm),
    P("Code, données et résultats : "
      "<font name='Mono' size='8.5'>github.com/mentalist011101/monoun-limnic-eruption-mining</font>",
      "formula"),
    PageBreak(),
]

# Résumé
story += [
    P("Résumé", "h1"),
    P(f"Le 15 août 1984, le lac Monoun (Cameroun) a libéré brutalement le CO<sub>2</sub> accumulé dans ses eaux "
      f"profondes, causant la mort de 37 personnes. Ce projet cherche, par extraction de règles d'association, "
      f"les combinaisons de conditions météorologiques qui ont précédé l'éruption, puis mesure leur fréquence "
      f"de réapparition entre 1983 et 2024. Les {N} jours de données NASA POWER sont discrétisés en 18 items, "
      f"chaque jour formant une transaction de 6 items, et l'algorithme FP-Growth est appliqué dans Weka "
      f"avec les critères de support et de confiance du cours."),
    P(f"Les facteurs « de risque » de l'énoncé (chute de pression, refroidissement nocturne, pluie extrême) "
      f"appartiennent à des saisons différentes et ne se combinent presque jamais : aucun jour ne les réunit tous "
      f"les trois sur 42 ans. Les sept jours précédant l'éruption ne présentent aucun item extrême ; leur signature "
      f"commune, {{{items_str(signature)}}}, est un motif fréquent de la mousson ordinaire "
      f"(support {fmt(sig_supp)}). Cette signature recule d'environ {fmt(-sig_trend.slope_days_per_decade, 0)} jours "
      f"par décennie, surtout après 2013. Ces résultats ne soutiennent pas l'hypothèse d'un déclenchement "
      f"météorologique et sont cohérents avec l'explication géologique retenue dans la littérature "
      f"(séisme et glissement de terrain sous-lacustre)."),
]

# 1. Introduction
story += [
    P("1. Introduction et problématique", "h1"),
    P("Le lac Monoun est un lac de cratère situé dans l'Ouest du Cameroun. Ses eaux profondes, anoxiques, "
      "accumulent du CO<sub>2</sub> d'origine magmatique. Le 15 août 1984, ce gaz s'est brusquement dégagé "
      "(éruption limnique) et a tué 37 personnes par asphyxie (Sigurdsson et al., 1987). Deux ans plus tard, "
      "un événement semblable au lac Nyos a fait plus de 1 700 victimes."),
    P("Une éruption limnique suppose que la stratification du lac soit rompue : les eaux profondes, chargées en "
      "gaz, remontent et dégazent. L'énoncé du projet fait l'hypothèse que des événements météorologiques "
      "composites ont pu provoquer cette rupture :"),
    *bullets([
        "une <b>chute de pression atmosphérique</b>, qui abaisse la solubilité du CO<sub>2</sub> (loi de Henry) ;",
        "un <b>refroidissement de surface</b> (nuits froides, pluie froide), qui rend l'eau de surface plus dense "
        "et favorise son plongement (inversion convective) ;",
        "des <b>pluies torrentielles</b> et un <b>vent fort</b>, qui brassent et déstabilisent la surface.",
    ]),
    P("Le projet poursuit deux objectifs :"),
    *bullets([
        "découvrir, par l'algorithme FP-Growth, les règles d'association entre conditions météorologiques "
        "au lac Monoun et identifier la « signature de risque » qui a précédé la catastrophe de 1984 ;",
        "évaluer la fréquence de réapparition de cette signature entre 1983 et 2024, dans le contexte du "
        "réchauffement climatique.",
    ]),
]

# 2. Données
desc = raw[["PRECTOTCORR", "T2M_MIN", "T2M_MAX", "DTR", "WS2M", "PS"]].describe().T
units = {"PRECTOTCORR": "mm/jour", "T2M_MIN": "°C", "T2M_MAX": "°C", "DTR": "°C", "WS2M": "m/s", "PS": "kPa"}
labels = {"PRECTOTCORR": "Précipitations corrigées", "T2M_MIN": "Température minimale à 2 m",
          "T2M_MAX": "Température maximale à 2 m", "DTR": "Amplitude thermique (calculée)",
          "WS2M": "Vitesse du vent à 2 m", "PS": "Pression de surface"}
rows = [["Variable", "Signification", "Unité", "Moyenne", "Min", "Médiane", "Max"]]
for v, d in desc.iterrows():
    rows.append([v, labels[v], units[v], fmt(d["mean"]), fmt(d["min"]), fmt(d["50%"]), fmt(d["max"])])
story += [
    P("2. Données", "h1"),
    P(f"Les données proviennent de NASA POWER (réanalyse MERRA-2), au point de coordonnées 5,58° N et "
      f"10,586° E, à 1 151 m d'altitude (moyenne de la maille). La série est journalière, du 1<super>er</super> "
      f"janvier 1983 au 31 décembre 2024, soit <b>{N} jours</b>. NASA POWER code les valeurs manquantes "
      f"par −999 : aucune n'est présente dans la série."),
    table(rows, [2.4 * cm, 4.6 * cm, 1.6 * cm, 2.1 * cm, 1.7 * cm, 2.1 * cm, 1.7 * cm]),
    P("Tableau 1 – Variables utilisées et statistiques descriptives (1983–2024).", "caption"),
    P("La variable <b>DOY</b> (jour de l'année, de 1 à 366) sert à définir la saison. L'amplitude thermique "
      "journalière DTR = T2M_MAX − T2M_MIN est calculée et arrondie à 2 décimales, comme les données sources."),
]

# 3. Préparation
disc = [["Variable", "Item", "Condition", "Jours (part)"]]
for var, its in [
    ("DOY (saison)", [("Season_Dry", "DOY ≥ 320 ou ≤ 60"), ("Season_Transition", "61 ≤ DOY ≤ 150"),
                      ("Season_Monsoon_Peak", "151 ≤ DOY ≤ 319")]),
    ("PRECTOTCORR", [("Rain_None", "< 1 mm"), ("Rain_Moderate", "1 – 20 mm"),
                     ("Rain_Extreme_Downpour", "> 20 mm")]),
    ("T2M_MIN", [("Tmin_Warm", "> 14 °C"), ("Tmin_Cold", "11 – 14 °C"), ("Tmin_Severe_Cooling", "< 11 °C")]),
    ("DTR", [("DTR_Suppressed", "< 8 °C"), ("DTR_Moderate", "8 – 14 °C"), ("DTR_Extreme", "> 14 °C")]),
    ("WS2M", [("Wind_Calm", "< 1,5 m/s"), ("Wind_Moderate", "1,5 – 3 m/s"), ("Wind_High_Shear", "> 3 m/s")]),
    ("PS", [("Pressure_High_Stable", "> 88,7 kPa"), ("Pressure_Normal", "88,4 – 88,7 kPa"),
            ("Pressure_Critical_Drop", "< 88,4 kPa")]),
]:
    for i, (it, cond) in enumerate(its):
        disc.append([var if i == 0 else "", it, cond, share(it)])
story += [
    P("3. Préparation des données", "h1"),
    P("3.1 Discrétisation", "h2"),
    P("Les règles d'association classiques portent sur des données binaires (présence ou absence d'un item). "
      "Chaque variable est donc découpée en trois classes selon les seuils de l'énoncé, utilisés tels quels. "
      "Chaque jour devient une <b>transaction</b> contenant exactement <b>6 items</b>, un par variable, "
      "sur 18 items possibles."),
    table(disc, [3.2 * cm, 5 * cm, 4.2 * cm, 4.2 * cm]),
    P("Tableau 2 – Discrétisation (seuils de l'énoncé) et répartition des items sur 1983–2024.", "caption"),
    P("3.2 Choix de traitement", "h2"),
    *bullets([
        "<b>Valeurs égales à un seuil.</b> Les intervalles de l'énoncé ne précisent pas les bornes. L'énoncé "
        "définit toujours les classes extrêmes par des inégalités strictes (< 1 mm, > 20 mm, > 14 °C, < 11 °C, "
        "< 88,4 kPa…) : une valeur égale à un seuil ne peut donc pas y appartenir. Elle est affectée à la "
        "<b>classe intermédiaire</b>, qui forme un intervalle fermé. Pour la pluie, cela rejoint la convention "
        "ETCCDI, selon laquelle un jour pluvieux reçoit au moins 1 mm. Seule divergence : l'ETCCDI compte "
        "20 mm comme « très fortes pluies » (R20mm : ≥ 20 mm) ; on suit l'énoncé, ce qui ne concerne que 2 jours.",
        "<b>Saison sèche.</b> Elle chevauche le changement d'année (DOY 320 à 060) et est traitée par une "
        "condition « DOY ≥ 320 ou DOY ≤ 60 ».",
        "<b>Années bissextiles.</b> Le décalage d'un jour des limites de saison après le 29 février est négligé.",
    ]),
    P("3.3 Encodage pour Weka", "h2"),
    P("Les transactions sont encodées en 18 attributs binaires {0, 1}. Chaque jour compte exactement six 1, "
      "ce que le code vérifie. Le fichier est exporté au format ARFF (<font name='Mono' size='8.5'>"
      "transactions_binary.arff</font>). Dans FP-Growth, l'option <font name='Mono' size='8.5'>positiveIndex = 2"
      "</font> indique que la valeur 1 signifie « item présent »."),
]

# 4. Méthode
story += [
    P("4. Méthode", "h1"),
    P("4.1 Critères de qualité", "h2"),
    P("On reprend les définitions du cours. Pour un motif M et une règle R = M<sub>1</sub> → M<sub>2</sub>, "
      "avec n le nombre total de transactions :"),
    P("supp(M) = n(M) / n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;supp(R) = supp(M<sub>1</sub> ∪ M<sub>2</sub>)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
      "conf(R) = supp(M<sub>1</sub> ∪ M<sub>2</sub>) / supp(M<sub>1</sub>)", "formula"),
    P("Le problème consiste à trouver toutes les règles dont le support et la confiance dépassent les seuils "
      "minSupp et minConf. L'anti-monotonie du support (si M<sub>1</sub> ⊆ M<sub>2</sub>, alors "
      "supp(M<sub>1</sub>) ≥ supp(M<sub>2</sub>)) a ici une conséquence directe : un motif qui contient un item "
      "de support 4 % a lui-même un support d'au plus 4 %. Les items rares ne peuvent donc apparaître que si "
      "minSupp est inférieur à leur propre support."),
    P("4.2 FP-Growth et Apriori", "h2"),
    P("Le cours détaille Apriori, qui suit l'approche « générer-et-élaguer ». L'énoncé impose FP-Growth "
      "(Han et al., 2000), qui relève de l'approche « diviser-pour-régner » : il compresse la base dans un arbre "
      "(FP-tree) et extrait les motifs fréquents sans générer de candidats. Pour les mêmes minSupp et minConf, "
      "les deux algorithmes produisent exactement les mêmes motifs fréquents et les mêmes règles ; seul le coût "
      "de calcul diffère."),
    KeepTogether([P("4.3 Paramètres Weka", "h2"), table([["Paramètre", "Valeur", "Rôle"],
           ["positiveIndex", "2", "La valeur 1 signifie « item présent »"],
           ["findAllRulesForSupportLevel", "True", "Toutes les règles au niveau minSupp (sinon Weka s'arrête à 10)"],
           ["lowerBoundMinSupport", "0,10 / 0,05 / 0,01", "minSupp (trois essais)"],
           ["metricType / minMetric", "Confidence / 0,7", "minConf, valeur utilisée dans le cours"],
           ["maxNumberOfItems", "−1", "Pas de limite sur la taille des motifs"]],
          [4.8 * cm, 3.6 * cm, 8.2 * cm]),
    P("Tableau 3 – Paramètres de FP-Growth dans Weka 3.8.", "caption")]),
    P("Commande Weka de l'essai principal :", "body"),
    P("weka.associations.FPGrowth -P 2 -I -1 -N 10 -T 0 -C 0.7 -D 0.05 -U 1.0 -M 0.01 -S", "code"),
    P("4.4 Sélection des règles", "h2"),
    P("Une confiance élevée ne suffit pas à rendre une règle intéressante. L'item Tmin_Warm est présent "
      f"{fmt(100 * item_share['Tmin_Warm'] / N, 1)} % des jours, ce qui est normal sous ce climat tropical "
      "d'altitude. Mais n'importe quelle règle X → Tmin_Warm atteint alors une confiance d'environ 0,95, "
      "simplement parce que l'item est presque toujours présent. Deux filtres sont donc appliqués après Weka :"),
    *bullets([
        "<b>règle informative</b> : conf(X → Y) > supp(Y), c'est-à-dire que la présence de X rend Y plus fréquent "
        "qu'en moyenne. L'écart conf(X → Y) − supp(Y) sert à classer les règles. Le rapport "
        "conf(X → Y) / supp(Y) correspond à la mesure appelée <i>lift</i> dans la littérature ; on n'utilise ici "
        "que les deux quantités définies dans le cours ;",
        "<b>règle non redondante</b> : on retire X → Y s'il existe X' ⊂ X tel que X' → Y ait une confiance au "
        "moins égale. Ajouter un item à la prémisse sans gagner en confiance n'apporte rien.",
    ]),
    P("4.5 Signature pré-éruption et réapparition", "h2"),
    P("L'éruption a eu lieu dans la nuit du 15 au 16 août 1984 (DOY 228). La fenêtre pré-éruption comprend les "
      "<b>7 jours</b> qui se terminent le jour de l'éruption (DOY 222 à 228). La <b>signature</b> est le motif "
      "présent tous les jours de cette fenêtre. Sa réapparition est mesurée par le nombre de jours par an qui "
      "la contiennent. Les tendances sont estimées par régression linéaire et testées par le test de "
      "Mann-Kendall (tau de Kendall entre l'année et le nombre de jours)."),
]

# 5. Résultats
story += [
    P("5. Résultats", "h1"),
    P("5.1 Vue d'ensemble des trois essais", "h2"),
    table([["minSupp", "Jours minimum", "Règles Weka", "Informatives", "Non redondantes"]] + summary,
          [2.8 * cm, 3.2 * cm, 3.2 * cm, 3.2 * cm, 3.6 * cm]),
    P("Tableau 4 – Nombre de règles par essai (minConf = 0,7).", "caption"),
    P("Les deux filtres retirent environ trois règles sur quatre. À minSupp = 0,10 et 0,05, <b>aucune règle ne "
      "contient d'item extrême</b>, conformément à l'anti-monotonie du support : les règles ne décrivent que la "
      "météo ordinaire. L'analyse porte donc sur l'essai à <b>0,01</b> (au moins 153 jours). Même à ce niveau, "
      "Tmin_Severe_Cooling (16 jours) et Wind_High_Shear (5 jours) ne peuvent apparaître dans aucune règle."),
    P("5.2 Règles impliquant les items extrêmes", "h2"),
]
rr = [["Prémisse X", "Conséquent Y", "supp", "conf", "supp(Y)", "conf − supp(Y)"]]
for it in EXT:
    rr.append([f"<b>{it}</b>", "", "", "", "", ""])
    rr += top_rules(it)
t = table(rr, [5.4 * cm, 5.2 * cm, 1.4 * cm, 1.3 * cm, 1.4 * cm, 1.9 * cm])
story += [
    t,
    P("Tableau 5 – Règles les plus marquées pour chaque item extrême (essai à 0,01, règles filtrées).", "caption"),
    P("Chaque facteur de risque de l'énoncé est associé à une saison différente :"),
    *bullets([
        "les <b>pluies extrêmes</b> vont avec le pic de mousson, un ciel couvert (DTR faible) et des nuits douces ;",
        "les <b>chutes de pression critiques</b> vont avec la saison de transition (mars à mai) ;",
        "les <b>nuits froides</b> et les <b>fortes amplitudes thermiques</b> vont avec la saison sèche, sans pluie.",
    ]),
    P("5.3 La signature composite de l'énoncé", "h2"),
    table([["Combinaison d'items", "Jours sur 1983–2024"]] + comp_rows, [11 * cm, 5 * cm]),
    P("Tableau 6 – Nombre de jours réunissant plusieurs facteurs de risque.", "caption"),
    P("La combinaison décrite par l'énoncé (chute de pression, choc thermique et pluie orageuse) n'apparaît "
      "<b>aucun jour</b> en 42 ans. Les paires sont elles-mêmes très rares."),
]

# 5.4 fenêtre
wr = [["Date", "Pluie (mm)", "Tmin (°C)", "DTR (°C)", "Vent (m/s)", "PS (kPa)", "Items"]]
wraw = raw[(raw["YEAR"] == 1984) & raw["DOY"].between(222, 228)]
months = {222: "9 août", 223: "10 août", 224: "11 août", 225: "12 août", 226: "13 août", 227: "14 août",
          228: "15 août"}
for (_, rw), (_, tr) in zip(wraw.iterrows(), window.iterrows()):
    its = [tr["Rain"], tr["DTR"], tr["Wind"]]
    wr.append([months[int(rw["DOY"])], fmt(rw["PRECTOTCORR"]), fmt(rw["T2M_MIN"]), fmt(rw["DTR"]),
               fmt(rw["WS2M"]), fmt(rw["PS"]), ", ".join(its)])
vr = [["Prémisse X", "Conséquent Y", "supp", "conf", "supp(Y)", "conf − supp(Y)"]]
vr += [[items_str(x.premise), items_str(x.consequent), fmt(x.supp, 3), fmt(x.conf), fmt(x.supp_consequent),
        fmt(x.gain, 3)] for x in verified.itertuples()]
story += [
    KeepTogether([
        P("5.4 La fenêtre pré-éruption et sa signature", "h2"),
        table(wr, [1.7 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 7.2 * cm]),
        P("Tableau 7 – Les 7 jours précédant l'éruption (1984). Tous les jours sont aussi Season_Monsoon_Peak, "
          "Tmin_Warm et Pressure_Normal.", "caption")]),
    P(f"<b>Aucun item extrême</b> n'apparaît dans la fenêtre : pluie modérée, nuits douces, ciel plutôt couvert, "
      f"vent calme et pression normale. La signature commune aux 7 jours est "
      f"<b>{{{items_str(signature)}}}</b>, avec un support de {fmt(sig_supp)} sur 1983–2024, soit environ "
      f"un jour sur quatre. Les règles filtrées que tous ces jours vérifient décrivent la mousson ordinaire :"),
    table(vr, [5.2 * cm, 5.4 * cm, 1.4 * cm, 1.3 * cm, 1.4 * cm, 1.9 * cm]),
    P("Tableau 8 – Règles (essai à 0,01, filtrées) vérifiées par les 7 jours de la fenêtre.", "caption"),
    P("Seules les deux premières règles apportent une information nette : en pic de mousson, la pluie est le "
      "plus souvent modérée. Les autres ont un écart conf − supp(Y) presque nul."),
]

# 5.5 réapparition
story += [
    P("5.5 Fréquence de réapparition de la signature", "h2"),
    figure(FIG / "signature_par_an.png",
           "Figure 1 – Nombre de jours par an présentant la signature pré-éruption (1984 en rouge)."),
    P(f"La signature apparaît en moyenne {fmt(sig_counts.mean(), 0)} jours par an. Elle devient <b>moins "
      f"fréquente</b> : pente de {fmt(sig_trend.slope_days_per_decade, 1)} jours par décennie, tendance "
      f"significative (Mann-Kendall : τ = {fmt(sig_trend.kendall_tau)}, p < 0,001). La baisse se concentre "
      f"après 2013 : {fmt(sig_before, 0)} jours par an en moyenne sur 1983–2013, contre {fmt(sig_after, 0)} "
      f"sur 2014–2024. L'année 1984 compte {int(sig_counts.loc[1984])} jours de signature, ce qui n'a rien "
      f"d'exceptionnel."),
    P("5.6 Évolution des facteurs extrêmes", "h2"),
    table([["Item", "Jours (total)", "Jours en 1984", "Pente (jours/décennie)", "p (Mann-Kendall)"]] + ext_trends,
          [4.8 * cm, 2.6 * cm, 2.6 * cm, 3.6 * cm, 3.0 * cm]),
    P("Tableau 9 – Tendances des items extrêmes sur 1983–2024.", "caption"),
    P(f"Au seuil de 5 %, les nuits froides et les chutes de pression critiques <b>diminuent</b>, tandis que les "
      f"fortes amplitudes thermiques <b>augmentent</b>. La hausse des pluies extrêmes sur l'année entière n'est "
      f"pas significative. La baisse des nuits froides est cohérente avec le réchauffement : la température "
      f"minimale moyenne passe de {fmt(tmin_80s, 1)} °C (1983–1989) à {fmt(tmin_20s, 1)} °C (2020–2024)."),
    P("5.7 Pourquoi la signature recule-t-elle après 2013 ?", "h2"),
    figure(FIG / "pression_mousson_par_an.png",
           "Figure 2 – Pression moyenne en pic de mousson (DOY 151–319) et seuil de 88,7 kPa."),
    P("Deux effets se combinent :"),
    *bullets([
        f"<b>la pression</b> : la moyenne de mousson passe de {fmt(ps_before, 3)} kPa (1983–2013) à "
        f"{fmt(ps_after, 3)} kPa (2014–2024), soit +{fmt(ps_after - ps_before, 2)} kPa, ce qui la place au "
        f"niveau du seuil de 88,7 kPa. De nombreux jours basculent de Pressure_Normal à Pressure_High_Stable "
        f"et sortent de la signature ;",
        f"<b>les pluies extrêmes de mousson</b> : elles passent de {fmt(rex_before, 1)} à {fmt(rex_after, 1)} "
        f"jours par an en moyenne. Ces jours remplacent Rain_Moderate et sortent eux aussi de la signature.",
    ]),
    P("L'effet de pression est un décalage très faible situé exactement sur un seuil : le comptage est donc "
      "sensible au choix de 88,7 kPa. La hausse des pluies extrêmes en mousson constitue un signal plus net."),
]

# 6. Discussion
story += [
    P("6. Discussion et limites", "h1"),
    P("Avec les seuils de l'énoncé, les règles d'association <b>ne soutiennent pas l'hypothèse</b> d'un "
      "déclenchement par un événement météorologique extrême composite. Les jours qui précèdent l'éruption sont "
      "des jours de mousson ordinaires, et la combinaison de facteurs décrite par l'énoncé n'existe pas dans les "
      "données. Ce résultat est cohérent avec l'explication retenue par Sigurdsson et al. (1987) : la "
      "stratification du lac aurait été rompue par un séisme et un glissement de terrain sous-lacustre survenus "
      "le 15 août, c'est-à-dire par un déclencheur interne au lac, que des données météorologiques ne peuvent "
      "pas détecter."),
    P("Plusieurs limites doivent être gardées à l'esprit :"),
    *bullets([
        "<b>Un seul événement.</b> La signature repose sur 7 jours d'un unique épisode : aucune validation "
        "statistique d'une « signature de risque » n'est possible avec un seul cas.",
        "<b>Données de réanalyse.</b> MERRA-2 fournit des moyennes sur une maille d'environ 50 km, qui lissent "
        "les extrêmes locaux (orages, rafales, refroidissement de la surface du lac).",
        f"<b>Seuils fixes.</b> Certains seuils ne correspondent pas aux centiles annoncés par l'énoncé : le P10 "
        f"réel de Tmin est de {fmt(p10_tmin, 1)} °C et non 11 °C, et le P95 de la pluie est de "
        f"{fmt(p95_rain, 1)} mm. Le seuil de pression de 88,7 kPa tombe sur la moyenne de mousson, ce qui rend "
        f"les comptages sensibles.",
        "<b>Co-occurrence et causalité.</b> Une règle d'association exprime une co-occurrence, pas une relation "
        "de cause à effet.",
        "<b>Tests de tendance.</b> Le test de Mann-Kendall suppose des années indépendantes ; une éventuelle "
        "autocorrélation n'est pas prise en compte.",
        "<b>Réapparition et risque.</b> Le lac Monoun est dégazé artificiellement depuis le début des années "
        "2000 : la réapparition d'une configuration météorologique ne signifie pas un risque équivalent "
        "à celui de 1984.",
    ]),
]

# 7. Conclusion
story += [
    P("7. Conclusion", "h1"),
    P("Ce projet a transformé 42 ans de données météorologiques journalières en une base transactionnelle de "
      f"{N} transactions et 18 items, puis en a extrait les règles d'association par FP-Growth avec les critères "
      "de support et de confiance. Les principaux résultats sont les suivants :"),
    *bullets([
        "les facteurs de risque de l'énoncé relèvent de saisons différentes (pluies extrêmes en mousson, chutes "
        "de pression en transition, nuits froides en saison sèche) et ne sont jamais réunis ;",
        f"la signature des 7 jours précédant l'éruption, {{{items_str(signature)}}}, est un motif fréquent de "
        f"la mousson ordinaire, sans aucun item extrême ;",
        f"cette signature recule d'environ {fmt(-sig_trend.slope_days_per_decade, 0)} jours par décennie, "
        "sous l'effet d'une légère hausse de la pression de mousson et de pluies extrêmes plus fréquentes "
        "en mousson ;",
        "les nuits froides diminuent et les fortes amplitudes thermiques augmentent, ce qui est cohérent avec "
        "le réchauffement.",
    ]),
    P("La catastrophe de 1984 ne semble donc pas avoir été précédée d'une configuration météorologique "
      "particulière. Deux prolongements seraient utiles : refaire l'analyse avec des seuils définis par les "
      "centiles réels de la série, et appliquer la même démarche au lac Nyos (21 août 1986) pour disposer d'un "
      "second événement."),
]

# Références
story += [
    P("Références", "h1"),
    *[P(r, "ref") for r in [
        "Agrawal, R., Imieliński, T. et Swami, A. (1993). Mining association rules between sets of items in "
        "large databases. <i>Proceedings of the ACM SIGMOD International Conference on Management of Data</i>, "
        "207–216.",
        "Han, J., Pei, J. et Yin, Y. (2000). Mining frequent patterns without candidate generation. "
        "<i>Proceedings of the ACM SIGMOD International Conference on Management of Data</i>, 1–12.",
        "Sigurdsson, H., Devine, J. D., Tchoua, F. M. et al. (1987). Origin of the lethal gas burst from Lake "
        "Monoun, Cameroun. <i>Journal of Volcanology and Geothermal Research</i>, 31(1–2). "
        "doi:10.1016/0377-0273(87)90002-3.",
        "ETCCDI – Expert Team on Climate Change Detection and Indices. Definitions of the 27 core indices. "
        "etccdi.pacificclimate.org/list_27_indices.shtml.",
        "NASA POWER – Prediction Of Worldwide Energy Resources, données journalières MERRA-2. power.larc.nasa.gov.",
        "Witten, I. H., Frank, E., Hall, M. A. et Pal, C. J. (2016). <i>Data Mining: Practical Machine Learning "
        "Tools and Techniques</i> (4<super>e</super> éd.). Morgan Kaufmann (logiciel Weka).",
        "Support de cours : Fouille de données – Règles d'association, M2 Data Science, ENSPY, 2026–2027.",
    ]],
]

# Annexe
story += [
    P("Annexe – Reproductibilité", "h1"),
    P("Le dépôt contient l'ensemble du code, des données et des résultats :"),
    P("data/raw/          données brutes NASA POWER<br/>"
      "data/processed/    transactions.csv, transactions_binary.arff, index.csv<br/>"
      "src/               preparation.py (phase 1), analyse.py (phase 3)<br/>"
      "notebooks/         01_preparation.ipynb, 02_analyse.ipynb<br/>"
      "weka/results/      sorties FP-Growth (minSupp 0,10 / 0,05 / 0,01)<br/>"
      "results/figures/   figures<br/>"
      "reports/           ce rapport et son script de génération", "code"),
    P("Ordre d'exécution : <font name='Mono' size='8.5'>01_preparation.ipynb</font>, puis FP-Growth dans Weka "
      "sur <font name='Mono' size='8.5'>transactions_binary.arff</font> (paramètres du tableau 3), puis "
      "<font name='Mono' size='8.5'>02_analyse.ipynb</font>."),
]


# ---------------------------------------------------------------- mise en page
def on_page(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "Projet 1 – Lac Monoun : règles d'association FP-Growth")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.setStrokeColor(colors.HexColor("#b8c4d2"))
    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2 * cm, bottomMargin=2 * cm,
                        title="Lac Monoun – Règles d'association FP-Growth", author="FOKOUO SAADIE Luciano")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(OUT)
