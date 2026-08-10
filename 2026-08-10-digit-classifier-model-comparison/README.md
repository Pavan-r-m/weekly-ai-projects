# Digit Classifier Model Comparison

A scikit-learn project that trains and fairly compares four very different
classification algorithms -- including a real neural network -- on the task
of recognizing handwritten digits (0-9), then digs into *why* the best model
still makes mistakes.

## What it does and why it's interesting

Most beginner ML tutorials train one model and report one accuracy number.
This project instead treats model selection like an actual ML workflow:

1. Trains **four qualitatively different algorithms** on the exact same
   train/test split: Logistic Regression (linear), an SVM with an RBF kernel
   (margin-based), a Random Forest (tree ensemble), and an `MLPClassifier`
   (a genuine feedforward neural network trained via backpropagation).
2. Compares them fairly on **accuracy, macro-F1, and training time** and
   picks a winner.
3. Analyzes the winner's mistakes with a **confusion matrix** and a
   **gallery of the actual misclassified digit images**.
4. As a bonus, visualizes the trained neural network's **first-layer
   weights** as 8x8 "feature detector" images -- a classic technique for
   building intuition about what a neural net's hidden units learn to
   respond to.

It runs entirely offline using scikit-learn's bundled `load_digits` dataset
(8x8 grayscale images, no downloads needed) in well under a minute on a
laptop CPU.

## Tech stack & key concepts

- **scikit-learn** — `LogisticRegression`, `SVC`, `RandomForestClassifier`,
  `MLPClassifier`, `StandardScaler`, `train_test_split`,
  `classification_report`, `confusion_matrix`
- **matplotlib** — grouped bar chart for model comparison, confusion matrix
  heatmap, misclassified-example gallery, neural network weight
  visualization
- Concepts: fair model benchmarking (same split, same metrics), linear vs.
  margin-based vs. ensemble vs. neural classifiers, macro-F1 for balanced
  multi-class evaluation, feature scaling, and interpreting a neural
  network's learned weights

## Installation

```bash
pip install -r requirements.txt
```

## How to run

```bash
python digit_classifier.py
```

This will:
1. Load and split the digits dataset (80/20, stratified)
2. Train all four models and print accuracy / macro-F1 / training time for
   each
3. Print a full classification report for the best-performing model
4. Save four PNGs to the current directory:
   - `model_comparison.png` — bar chart of accuracy & F1 across models
   - `confusion_matrix.png` — confusion matrix for the best model
   - `misclassified.png` — gallery of digits the best model got wrong
   - `mlp_hidden_units.png` — visualization of the neural net's learned
     first-layer weights

## Example output

```
Loading data (sklearn digits: 8x8 grayscale, 10 classes)...
Train samples: 1437 | Test samples: 360

Training & evaluating models...

Logistic Regression    | accuracy=0.9722 | macro_f1=0.9719 | train_time=0.279s
SVM (RBF kernel)       | accuracy=0.9806 | macro_f1=0.9805 | train_time=0.038s
Random Forest          | accuracy=0.9694 | macro_f1=0.9689 | train_time=0.391s
Neural Net (MLP)       | accuracy=0.9667 | macro_f1=0.9664 | train_time=0.467s

Best model: SVM (RBF kernel) (accuracy=0.9806)

Classification report for SVM (RBF kernel):
              precision    recall  f1-score   support
           0      1.000     1.000     1.000        36
           1      0.972     0.972     0.972        36
           ...
    accuracy                          0.981       360
```

(Exact numbers can vary slightly depending on scikit-learn version, but all
four models consistently land in the 96-98% accuracy range on this dataset.)

## How it works

1. **Data prep**: `load_digits()` returns 1,797 samples of 8x8 pixel images
   (values 0-16), flattened to 64-dim feature vectors. Features are
   standardized (zero mean, unit variance) for the models that benefit from
   it (logistic regression, SVM, MLP); the tree-based Random Forest uses raw
   pixel values since it's scale-invariant.
2. **Model zoo**: four algorithms are trained on the identical train/test
   split so comparisons are apples-to-apples:
   - *Logistic Regression* — a linear decision boundary per class
   - *SVM (RBF kernel)* — finds max-margin boundaries in a
     higher-dimensional kernel space, historically very strong on small
     image datasets like this one
   - *Random Forest* — an ensemble of 300 decision trees voting together
   - *MLPClassifier* — a 2-hidden-layer (64, 32) feedforward neural network
     trained with the Adam optimizer via backpropagation
3. **Fair comparison**: each model reports test-set accuracy, macro-F1
   (which weights all 10 classes equally regardless of support), and wall
   clock training time, printed side by side.
4. **Error analysis**: the best model's confusion matrix reveals which
   digit pairs it confuses most (commonly 4/9, 3/8, 7/9), and the
   misclassified gallery shows the literal images it got wrong.
5. **Interpretability bonus**: the MLP's first weight matrix
   (`coefs_[0]`, shape 64 x n_hidden_units) is reshaped column-by-column
   back into 8x8 images. Each one shows what pixel pattern a given hidden
   neuron has learned to respond most strongly to -- a simple but genuine
   window into what the neural network "sees."

No API key needed -- this project uses no external LLM/API calls at all.
