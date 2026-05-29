"""
Fake News Detector — Logistic Regression + TF-IDF
==================================================
Classifies news headlines/snippets as REAL or FAKE using classic NLP
techniques: TF-IDF vectorisation + Logistic Regression.

A synthetic dataset is bundled inside this script so the project runs
entirely offline without any external downloads.

Usage:
    python fake_news_detector.py          # train, evaluate, and demo
    python fake_news_detector.py --text "Your headline here"
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Synthetic dataset (60 REAL + 60 FAKE headlines)
# ---------------------------------------------------------------------------
REAL_HEADLINES = [
    "Scientists confirm new treatment reduces cancer risk by 30 percent in trial",
    "Federal Reserve raises interest rates by quarter point amid inflation concerns",
    "Earthquake measuring 6.2 strikes off coast of Japan no tsunami warning issued",
    "Study finds regular exercise linked to improved mental health outcomes",
    "Government announces new infrastructure spending plan worth 1.2 trillion dollars",
    "Tech company reports quarterly earnings above analyst expectations",
    "WHO recommends updated COVID-19 vaccine formulation for winter season",
    "United Nations releases annual climate change progress report",
    "Local elections see record voter turnout across three major cities",
    "NASA confirms successful landing of new Mars rover after seven-month journey",
    "Researchers develop biodegradable plastic alternative from seaweed extract",
    "Central bank holds interest rates steady amid mixed economic signals",
    "New highway connecting two major cities opens ahead of schedule",
    "Hospital network adopts AI-assisted diagnostic tool for radiology",
    "Trade agreement between two nations enters force after ratification",
    "Scientists discover new species of deep-sea fish near Pacific ridge",
    "University study links sleep deprivation to higher cardiovascular risk",
    "Stock markets close higher following positive jobs report",
    "International climate summit reaches agreement on carbon reduction targets",
    "Drug regulator approves new migraine medication after successful trials",
    "City council votes to expand public transit network by 2028",
    "Tech giant announces layoffs affecting 5 percent of global workforce",
    "Archaeologists uncover 2000-year-old mosaic in southern Italy",
    "Report shows renewable energy now accounts for 40 percent of US electricity",
    "Census bureau releases preliminary population data for major metropolitan areas",
    "Airlines report strong summer travel demand as bookings hit record levels",
    "New study shows Mediterranean diet associated with lower dementia risk",
    "Government watchdog flags procurement irregularities in defence contracts",
    "Astronomers detect gravitational waves from rare neutron star collision",
    "School district rolls out updated math curriculum across all grade levels",
    "Consumer prices rose 0.2 percent last month core inflation stays steady",
    "Biologists document wolf pack recolonisation in northern highlands region",
    "State legislature passes bill expanding broadband access in rural areas",
    "Pharmaceutical firm recalls batch of blood pressure medication over contamination",
    "Record rainfall causes flooding across south-eastern coastal communities",
    "Diplomatic talks resume between two countries after six-month pause",
    "Automaker recalls 120000 vehicles over potential brake system defect",
    "New research finds urban green spaces reduce heat island effect significantly",
    "Court rules in favour of plaintiffs in landmark data privacy lawsuit",
    "Utility company commits to phasing out coal plants by 2035",
    "Pediatric hospital reports decline in childhood obesity rates over decade",
    "National park service reports record 330 million visits in calendar year",
    "Researchers identify gene variant associated with higher longevity",
    "Central government releases draft regulations on autonomous vehicle testing",
    "Economists revise GDP growth forecast upward to 2.8 percent for year",
    "Committee recommends stricter emissions standards for heavy-duty trucks",
    "New satellite launched to monitor deforestation in Amazon basin",
    "Study confirms long-term safety of mRNA vaccine technology in adults",
    "Shipping company deploys first fleet of hydrogen-powered cargo vessels",
    "Municipal government introduces composting programme to cut landfill waste",
    "Researchers develop low-cost water purification filter using biochar",
    "Airlines ordered to improve compensation process for cancelled flights",
    "Antarctic ice sheet survey reveals slower melt rate than previous models",
    "Technology standards body finalises next-generation WiFi specification",
    "Conservation programme reports humpback whale population recovery",
    "Election commission certifies results after audit confirms vote totals",
    "New building code requires solar-ready wiring in residential construction",
    "Scientists sequence genome of endangered Sumatran rhino subspecies",
    "Central bank digital currency pilot enters second phase of testing",
    "Health ministry updates nutritional guidelines recommending less added sugar",
]

FAKE_HEADLINES = [
    "EXPOSED Government secretly putting mind-control chemicals in tap water CONFIRMED",
    "Doctors REFUSE to tell you this one weird trick that cures all cancers overnight",
    "BREAKING Reptilian elites caught shapeshifting on live television footage",
    "Bill Gates personally admits to depopulation agenda in leaked private video",
    "5G towers proven to cause COVID-19 according to banned whistleblower scientist",
    "NASA insider reveals Moon landing was filmed in Hollywood studio by Stanley Kubrick",
    "Deep state operatives poisoning food supply to make population obedient slaves",
    "Miracle herb suppressed by Big Pharma reverses diabetes in just 48 hours",
    "SHOCKER Vaccines contain microchips that activate at 60 GHz frequency",
    "World leaders gather in secret bunker to plan global currency collapse",
    "Ancient alien technology discovered under Antarctic ice kept hidden from public",
    "Mainstream media BLACKOUT as president signs secret martial law executive order",
    "Doctors paid millions to hide cure for Alzheimer's sitting in your kitchen",
    "Pentagon whistleblower UFO wreckage stored in classified Nevada facility CONFIRMED",
    "Chemtrails contain DNA-altering nanoparticles says fired air force colonel",
    "Banker admits in deathbed confession that all elections are rigged by globalists",
    "This common household spice eliminates cancer cells says banned Harvard study",
    "LEAK United Nations plan to replace world population with robots by 2030",
    "Soros funded group paying millions to actors to fake climate change protests",
    "Secret court documents reveal every politician is blackmailed by shadow government",
    "Fluoride in water proven to lower IQ and make citizens easier to control",
    "Hollywood celebrity satanic rituals exposed in EXCLUSIVE insider footage WATCH NOW",
    "FDA secretly approved a chip implant that reads your thoughts without consent",
    "Proof that mainstream news anchors all read from the same script written by CIA",
    "Elon Musk reveals in coded tweet that moon is actually a hologram projector",
    "This doctor was SILENCED after discovering that sugar causes no health issues",
    "Mass graves found near major city hushed up by corrupt officials say insiders",
    "CONFIRMED Biological weapons laboratory operating under major fast-food chain",
    "Central banks plan to ban cash by 2025 to track every penny you spend forever",
    "Whistleblower reveals sunscreen causes cancer and is designed to keep you sick",
    "World War 3 already started and mainstream media covering it up right now",
    "New world order officially launches secret digital ID chip programme globally",
    "Military general breaks silence on government weather modification programme",
    "Mega corporation pays scientists to fake global warming data reveals insider",
    "SCANDAL Childrens vitamins contain sterilisation agents say banned scientists",
    "Anonymous group hacks government server reveals list of cloned politicians",
    "Secret treaty signed by 50 nations to surrender sovereignty to world government",
    "This suppressed technology would give everyone free energy but oil companies blocked it",
    "Population control drug secretly added to restaurant food across 40 countries",
    "EXCLUSIVE Leaked memo shows plan to trigger financial crash before election",
    "Entire media industry owned by just three families who control all narratives",
    "Top virologist fired after proving COVID was engineered as biological weapon",
    "Prince admits in private recording that royal family practices blood rituals",
    "Google algorithm intentionally showing you propaganda to alter your political views",
    "Underground city built beneath capital for elites to survive engineered disaster",
    "Harvard professor silenced for proving that vaccines cause autism in new study",
    "Government drone swarms used to spy on citizens in their own homes REVEALED",
    "Banned documentary exposes that dinosaurs never existed and fossils are planted",
    "Deep state planning to trigger nuclear false flag to justify martial law takeover",
    "Retired CIA operative confirms all terrorist attacks since 2001 were inside jobs",
    "Nano-robots in COVID vaccine activate to harvest your biological data wirelessly",
    "Mainstream scientist admits privately that evolution is a hoax to hide the truth",
    "Secret ingredient in fast food causes addiction stronger than heroin say researchers",
    "Shadow government controls all 195 nations through debt slavery system EXPOSED",
    "Top health official reveals annual flu shot intentionally weakens your immune system",
    "Banker whistleblower cryptocurrency ban planned to force everyone onto CBDC chip",
    "Military insiders confirm alien invasion agreement signed at Area 51 decades ago",
    "This one food combination cures all autoimmune diseases says fired immunologist",
    "LEAKED Government roadmap to introduce social credit score in Western nations",
    "Famous scientist reveals on deathbed that Einstein stole all his theories from others",
]


def build_dataset() -> pd.DataFrame:
    """Combine REAL and FAKE headlines into a labelled DataFrame."""
    real_df = pd.DataFrame({"text": REAL_HEADLINES, "label": 1, "label_name": "REAL"})
    fake_df = pd.DataFrame({"text": FAKE_HEADLINES, "label": 0, "label_name": "FAKE"})
    df = pd.concat([real_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/numbers, strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Model pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Return a TF-IDF + Logistic Regression sklearn Pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),   # unigrams + bigrams capture phrase patterns
            max_features=5000,
            sublinear_tf=True,    # log(1+tf) dampens high-frequency terms
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, X_test, y_test) -> None:
    """Print metrics and save evaluation plots."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 55)
    print("CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score : {auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["FAKE", "REAL"],
    ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, label=f"Logistic Reg (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out_path = Path("evaluation_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved → {out_path}")


def show_top_features(pipeline: Pipeline, n: int = 12) -> None:
    """Print the TF-IDF features most predictive of FAKE vs REAL."""
    vectoriser = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectoriser.get_feature_names_out())
    coef = clf.coef_[0]

    top_fake_idx = coef.argsort()[:n]
    top_real_idx = coef.argsort()[-n:][::-1]

    print("\n" + "=" * 55)
    print(f"TOP {n} FEATURES → FAKE")
    print("=" * 55)
    for i in top_fake_idx:
        print(f"  {feature_names[i]:<35s}  coef={coef[i]:+.3f}")

    print("\n" + "=" * 55)
    print(f"TOP {n} FEATURES → REAL")
    print("=" * 55)
    for i in top_real_idx:
        print(f"  {feature_names[i]:<35s}  coef={coef[i]:+.3f}")


# ---------------------------------------------------------------------------
# 5. Demo
# ---------------------------------------------------------------------------

DEMO_TEXTS = [
    "Scientists discover potential link between gut bacteria and depression",
    "EXPOSED Secret government mind control programme uses 5G towers nationwide",
    "Federal court rules social media company violated antitrust law",
    "Doctors REFUSE to reveal this one natural trick that cures all disease instantly",
    "New battery technology could double the range of electric vehicles by 2028",
    "SHOCKING Reptilian shapeshifters control all world governments CONFIRMED",
]


def run_demo(pipeline: Pipeline) -> None:
    """Classify a handful of demo headlines."""
    print("\n" + "=" * 55)
    print("DEMO PREDICTIONS")
    print("=" * 55)
    for text in DEMO_TEXTS:
        label = pipeline.predict([text])[0]
        prob = pipeline.predict_proba([text])[0]
        verdict = "REAL ✓" if label == 1 else "FAKE ✗"
        display = text[:65] + "..." if len(text) > 65 else text
        print(f"\n  Text   : {display}")
        print(f"  Verdict: {verdict}  (FAKE={prob[0]:.2%}, REAL={prob[1]:.2%})")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("--text", type=str, default=None,
                        help="Classify a single custom headline.")
    args = parser.parse_args()

    print("=" * 55)
    print(" FAKE NEWS DETECTOR — TF-IDF + Logistic Regression")
    print("=" * 55)

    df = build_dataset()
    print(f"\nDataset : {len(df)} headlines  |  "
          f"REAL={df['label'].sum()}  FAKE={(df['label']==0).sum()}")

    X = df["text"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train   : {len(X_train)}   Test : {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    if args.text:
        label = pipeline.predict([args.text])[0]
        prob = pipeline.predict_proba([args.text])[0]
        verdict = "REAL" if label == 1 else "FAKE"
        print(f"\nInput  : {args.text}")
        print(f"Verdict: {verdict}  (FAKE={prob[0]:.2%}, REAL={prob[1]:.2%})")
    else:
        evaluate(pipeline, X_test, y_test)
        show_top_features(pipeline)
        run_demo(pipeline)
        model_path = Path("fake_news_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"\nModel saved → {model_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
