# Active Learning: Uncertainty Sampling vs. Random Sampling

Labeling data is often the most expensive part of a machine learning
project — someone has to read every contract, look at every scan, or
tag every support ticket by hand. **Active learning** is the idea that a
model can help pick which examples are worth labeling next, so you reach
good accuracy with a fraction of the labels a random sample would need.

This project builds a pool-based active learning loop from scratch and
runs a head-to-head experiment: does asking the model "which examples are
you least sure about?" actually beat labeling examples at random, given
the exact same budget?

## Why it's interesting

- It's a real, measurable claim, not a vibe: the script plots test
  accuracy vs. number of labeled examples for both strategies on the
  *same* dataset, *same* model, *same* random seed for the initial split —
  the only difference is which points get labeled each round.
- Uses **margin sampling** (the gap between a model's top two predicted
  class probabilities) as the uncertainty measure — simple to implement,
  no extra dependencies, and a standard technique used in real active
  learning systems (e.g. data labeling pipelines at ML companies).
- Directly answers a practical question: "how many fewer labels do I need
  to hit 90% accuracy if I'm smart about what I label?" On the breast
  cancer dataset in the example run below, uncertainty sampling reached
  90% accuracy with **20 labels** vs. **60 labels** for random sampling —
  a 3x reduction in labeling effort.

## Tech stack & key concepts

- **scikit-learn** — `RandomForestClassifier` as the underlying model,
  plus the built-in `breast_cancer` and `digits` toy datasets (no
  downloads or API keys needed).
- **NumPy** — uncertainty scoring and pool bookkeeping.
- **Matplotlib** — learning curve visualization.
- **Key concept: pool-based active learning** — start with a tiny labeled
  "seed" set and a large pool of unlabeled points; each round, the
  strategy under test picks a batch of unlabeled points to "label"
  (in this simulation, we already know their true labels and just reveal
  them), retrain, and measure test accuracy.
- **Key concept: margin sampling** — uncertainty score = `-(top1_prob -
  top2_prob)`. A small margin between the top two classes means the model
  is on the fence about that example, which makes it a high-value label
  to acquire.

## Installation

```bash
pip install -r requirements.txt
```

## How to run

Default run (breast cancer dataset):

```bash
python active_learning.py
```

Try the digits dataset (10-class, higher-dimensional) with a bigger budget:

```bash
python active_learning.py --dataset digits --n-initial 20 --n-query 10 --n-rounds 25
```

All options:

```
--dataset {breast_cancer,digits}   Which built-in dataset to use (default: breast_cancer)
--n-initial N                      Size of the initial random "seed" labeled set (default: 10)
--n-query N                        How many new labels to acquire per round (default: 5)
--n-rounds N                       Number of query rounds to run (default: 20)
--seed N                           Random seed for reproducibility (default: 42)
--output-dir DIR                   Where to write the CSV and PNG outputs (default: current dir)
```

The script writes two files: `active_learning_results.csv` (raw
accuracy-vs-labels data for both strategies) and `learning_curves.png`
(the comparison plot).

## Example output

```
Loading dataset: breast_cancer
Train pool size: 398   Test set size: 171
Seed labels: 10   Query batch: 5   Rounds: 15

Running strategy: random ...
  -> final: 85 labels, test accuracy = 0.9357
Running strategy: uncertainty ...
  -> final: 85 labels, test accuracy = 0.9474

Labels needed to first reach various accuracy targets:
  accuracy >= 0.80:  random needs   10 labels,  uncertainty needs   10 labels
  accuracy >= 0.84:  random needs   10 labels,  uncertainty needs   10 labels
  accuracy >= 0.86:  random needs   20 labels,  uncertainty needs   15 labels
  accuracy >= 0.88:  random needs   30 labels,  uncertainty needs   15 labels
  accuracy >= 0.90:  random needs   60 labels,  uncertainty needs   20 labels
  accuracy >= 0.92:  random needs   75 labels,  uncertainty needs   20 labels

Saved raw results to ./active_learning_results.csv
Saved learning curve plot to ./learning_curves.png
```

The uncertainty-sampling curve consistently sits above (or reaches
targets earlier than) the random-sampling curve, especially once the
model has enough seed examples to have a meaningful decision boundary to
be uncertain about.

## How it works

1. **Split the data.** A held-out test set is carved out up front and
   never touched by either labeling strategy — it's only used to measure
   accuracy at each round.
2. **Seed round.** Both strategies start from the *same* small randomly
   chosen labeled set (same seed = same starting point, so the comparison
   is fair).
3. **Each round:**
   - Train a `RandomForestClassifier` on everything labeled so far.
   - Score it on the held-out test set and record `(n_labeled, accuracy)`.
   - Pick the next batch of points to label from the unlabeled pool:
     - *Random strategy:* pick uniformly at random.
     - *Uncertainty strategy:* run `predict_proba` on the unlabeled pool,
       compute the margin between the top two class probabilities for
       each point, and pick the points with the **smallest** margin
       (i.e. the model is most torn between two classes).
   - Move the newly "labeled" points from the unlabeled pool into the
     labeled set.
4. **Repeat** for a fixed number of rounds, then plot both strategies'
   learning curves on the same axes and report how many labels each
   strategy needed to cross several accuracy thresholds.

No API keys, internet access, or GPU required — everything runs on
CPU in a few seconds using scikit-learn's built-in toy datasets.
