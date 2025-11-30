from concurrent.futures import ProcessPoolExecutor
import itertools
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import scipy
from scipy.sparse import csr_matrix
from sklearn import naive_bayes
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import time
import warnings

def read_and_process(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        review= f.read()
    tokens= nltk.word_tokenize(review)
    pos_tags= nltk.pos_tag(tokens)
    tagged_review= " ".join([f"{word}_{tag}" for word, tag in pos_tags])
    return tagged_review

def get_data(directory_path):
    train_dir= os.path.join(directory_path, "train")
    test_dir= os.path.join(directory_path, "test")

    pos_files= [os.path.join(train_dir, "pos", file) for file in os.listdir(os.path.join(train_dir, "pos"))] + [os.path.join(test_dir, "pos", file) for file in os.listdir(os.path.join(test_dir, "pos"))]
    neg_files= [os.path.join(train_dir, "neg", file) for file in os.listdir(os.path.join(train_dir, "neg"))] + [os.path.join(test_dir, "neg", file) for file in os.listdir(os.path.join(test_dir, "neg"))]

    all_files= pos_files + neg_files
    labels= [1] * len(pos_files) + [0] * len(neg_files)

    with ProcessPoolExecutor() as executor:
        reviews= list(executor.map(read_and_process, all_files))

    return reviews, labels

def extract_all_features(reviews):
    tfidf_vectorizer= TfidfVectorizer(max_features=5000)
    tfidf_features= tfidf_vectorizer.fit_transform(reviews)

    cooccurrence_vectorizer= CountVectorizer(ngram_range=(2, 2), max_features=5000)
    cooccurrence_features= cooccurrence_vectorizer.fit_transform(reviews)

    mean_word_lengths= np.array([
        np.mean([len(word) for word in nltk.word_tokenize(review)]) for review in reviews
    ]).reshape(-1, 1)
    mean_word_lengths= csr_matrix(mean_word_lengths)

    avg_sentence_lengths= np.array([
        np.mean([len(nltk.word_tokenize(sentence)) for sentence in nltk.sent_tokenize(review)]) for review in reviews
    ]).reshape(-1, 1)
    avg_sentence_lengths= csr_matrix(avg_sentence_lengths)

    return tfidf_features, cooccurrence_features, mean_word_lengths, avg_sentence_lengths

def naive_bayes_model(X_train, y_train, X_test, y_test):
    model= naive_bayes.MultinomialNB()
    start_time= time.time()
    model.fit(X_train, y_train)
    predictions= model.predict(X_test)
    end_time= time.time()
    total_time= end_time - start_time
    accuracy= accuracy_score(y_test, predictions)
    precision= precision_score(y_test, predictions, zero_division=0)
    recall= recall_score(y_test, predictions)
    confusion= confusion_matrix(y_test, predictions)
    return accuracy, precision, recall, confusion, total_time

def logistic_regression_model(X_train, y_train, X_test, y_test):
    model= LogisticRegression(max_iter=2000, solver="saga")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category= ConvergenceWarning)
        start_time= time.time()
        model.fit(X_train, y_train)
        predictions= model.predict(X_test)
        end_time= time.time()
    total_time= end_time - start_time
    accuracy= accuracy_score(y_test, predictions)
    precision= precision_score(y_test, predictions, zero_division=0)
    recall= recall_score(y_test, predictions)
    confusion= confusion_matrix(y_test, predictions)
    return accuracy, precision, recall, confusion, total_time

def show_results(name, accuracy, precision, recall, confusion_matrix, time_taken):
    print(name, "Model", "\n")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f1_score= 2 * (precision*recall) / (precision+recall)
    print("F1 Score:", f1_score)
    print("Confusion Matrix:")
    print(tabulate(confusion_matrix, tablefmt="fancy_grid"))
    print("Time Taken:", time_taken, "seconds\n")

def plot_results(results):
    feature_labels= ["TF-IDF", "Co-occurrence", "Word Length", "Sentence Length"]
    f1_scores, precisions, recalls, times, labels= [], [], [], [], []

    for result in results:
        features= result["features"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nb_f1= 2 * (result["nb"][1] * result["nb"][2]) / (result["nb"][1] + result["nb"][2])
            lr_f1= 2 * (result["lr"][1] * result["lr"][2]) / (result["lr"][1] + result["lr"][2])
        f1_scores.append((nb_f1, lr_f1))
        precisions.append((result["nb"][1], result["lr"][1]))
        recalls.append((result["nb"][2], result["lr"][2]))
        times.append((result["nb"][3], result["lr"][3]))
        labels.append(", ".join([label for label, use in zip(feature_labels, features) if use]) or "None")

    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    x= np.arange(len(labels))
    width= 0.35

    fig, ax= plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, [f1[0] for f1 in f1_scores], width, label="NB F1 Score", color="orange")
    ax.bar(x + width/2, [f1[1] for f1 in f1_scores], width, label="LR F1 Score", color="red")
    ax.set_ylabel("F1 Scores")
    ax.set_xlabel("Feature Combinations")
    ax.set_title("Model F1 Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "model_f1_scores.png"))

    fig, ax= plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, [prec[0] for prec in precisions], width, label="NB Precision", color="orange")
    ax.bar(x + width/2, [prec[1] for prec in precisions], width, label="LR Precision", color="red")
    ax.set_ylabel("Precision")
    ax.set_xlabel("Feature Combinations")
    ax.set_title("Model Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "model_precision.png"))

    fig, ax= plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, [rec[0] for rec in recalls], width, label="NB Recall", color="orange")
    ax.bar(x + width/2, [rec[1] for rec in recalls], width, label="LR Recall", color="red")
    ax.set_ylabel("Recall")
    ax.set_xlabel("Feature Combinations")
    ax.set_title("Model Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "model_recall.png"))

    fig, ax= plt.subplots(figsize=(12, 8))
    ax.bar(x - width/2, [time[0] for time in times], width, label="NB Time Taken", color="orange")
    ax.bar(x + width/2, [time[1] for time in times], width, label="LR Time Taken", color="red")
    ax.set_ylabel("Time Taken (seconds)")
    ax.set_xlabel("Feature Combinations")
    ax.set_title("Model Time Taken")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", "model_time.png"))

def main():
    print("Loading Data...", "\n")
    directory_path= "aclImdb"
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    reviews, labels= get_data(directory_path)

    print("Extracting Features...", "\n")
    tfidf_features, cooccurrence_features, word_length_features, sentence_length_features= extract_all_features(reviews)

    feature_combinations= [
        combo for combo in itertools.product([True, False], repeat=4) if any(combo)
    ]
    results= []

    print("Results: ")
    print("#####################################################################################################", "\n")
    for use_tfidf, use_cooccurrence, use_word_length, use_sentence_length in feature_combinations:
        print(f"Evaluating Features: TF-IDF= {use_tfidf}, Co-occurrence= {use_cooccurrence}, Word Length= {use_word_length}, Sentence Length= {use_sentence_length}", "\n")
        features= []
        if use_tfidf:
            features.append(tfidf_features)
        if use_cooccurrence:
            features.append(cooccurrence_features)
        if use_word_length:
            features.append(word_length_features)
        if use_sentence_length:
            features.append(sentence_length_features)

        if features:
            X= scipy.sparse.hstack(features)
        else:
            X= np.zeros((len(reviews), 1))

        y= np.array(labels)
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

        nb_accuracy, nb_precision, nb_recall, nb_confusion, nb_time= naive_bayes_model(X_train, y_train, X_test, y_test)
        lr_accuracy, lr_precision, lr_recall, lr_confusion, lr_time= logistic_regression_model(X_train, y_train, X_test, y_test)

        results.append({
            "features": (use_tfidf, use_cooccurrence, use_word_length, use_sentence_length),
            "nb": (nb_accuracy, nb_precision, nb_recall, nb_time),
            "lr": (lr_accuracy, lr_precision, lr_recall, lr_time)
        })

        show_results("Naive Bayes", nb_accuracy, nb_precision, nb_recall, nb_confusion, nb_time)
        show_results("Logistic Regression", lr_accuracy, lr_precision, lr_recall, lr_confusion, lr_time)
        print("#####################################################################################################", "\n")

    plot_results(results)

if __name__== "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
