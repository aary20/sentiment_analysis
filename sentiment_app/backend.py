import time
import string
import numpy as np
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import os
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer  # Added import
from sklearn.naive_bayes import MultinomialNB                # Added import

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download("wordnet")

porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stopwords = stopwords.words('english')

models = ("BernoulliNB",
          "Logistic Regression",
          "GradientBoostingClassifier",
          "LinearSVC",
          'Distilbert Pipeline')


def load_dataset():
    """
    Loads a dataset from CSV and returns it as a Pandas DataFrame.
    Supports:
    - custom_text.csv/custom_texts.csv with columns like text,label
    - amazon_reviews.csv (legacy: verified_reviews,feedback)
    """
    # Prefer custom dataset
    custom_paths = [
        "custom_text.csv",
        "custom_texts.csv",
        os.path.join("data", "custom_text.csv"),
        os.path.join("data", "custom_texts.csv"),
    ]
    df = None
    for p in custom_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        # Fallback to legacy Amazon dataset
        legacy_paths = [
            "amazon_reviews.csv",
            os.path.join("data", "amazon_reviews.csv"),
        ]
        for p in legacy_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
    if df is None:
        raise FileNotFoundError("No dataset found. Place custom_text.csv or amazon_reviews.csv.")

    # Normalize to text + label columns
    text_col = "text" if "text" in df.columns else ("verified_reviews" if "verified_reviews" in df.columns else None)
    label_col = "label" if "label" in df.columns else ("feedback" if "feedback" in df.columns else None)
    if text_col is None or label_col is None:
        raise ValueError("Dataset must have text and label columns (e.g., text,label or verified_reviews,feedback).")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"].replace("", np.nan, inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)

    # Map labels to 0/1 for consistency
    def map_label(v):
        if isinstance(v, str):
            l = v.strip().lower()
            if l in ["positive", "pos", "1", "true", "yes"]:
                return 1
            if l in ["negative", "neg", "0", "false", "no"]:
                return 0
            try:
                return 1 if int(l) == 1 else 0
            except Exception:
                return 1 if "pos" in l or "good" in l or "like" in l else 0
        return int(v)
    df["label"] = df["label"].apply(map_label).astype(int)
    return df


def get_wordnet_pos(treebank_tag):
    """
    Map POS tag from the Treebank format to the WordNet format for WordNetLemmatizer.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def model_evaluate(model, x_train, x_test, y_train, y_test):
    """
    Evaluate the performance of a machine learning model on a test set.
    """
    y_pred = model.predict(x_test)
    classification_dict = classification_report(y_test, y_pred, output_dict=True)
    try:
        pr_train = model.predict_proba(x_train)[:, 1]
        pr_test = model.predict_proba(x_test)[:, 1]
        train_auc = roc_auc_score(y_train, pr_train)
        test_auc = roc_auc_score(y_test, pr_test)
    except AttributeError as e:
        train_auc = None
        test_auc = None
    return classification_dict, train_auc, test_auc


class LemmaTokenizer:
    """
    Custom tokenizer class that uses lemmatization to process text.
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag))
                 for word, tag in words_and_tags]


def preprocessing(message):
    """
    This function preprocesses a given message by removing punctuation, stop words, and non-alphabetic characters.
    """
    test_punc_removed = [word for word in message if word not in string.punctuation]
    test_punc_removed_join = ''.join(test_punc_removed)
    test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords and word.lower().isalpha() and word is not None]
    test_punc_removed_join_clean = ' '.join(test_punc_removed_join_clean)
    return test_punc_removed_join_clean


def feature_engineering(data_df):
    """
    Preprocesses the text data and performs feature engineering by oversampling,
    splitting the data into train and test sets, and vectorizing the text data using
    TF-IDF with n-grams.
    """
    X = data_df["text"]
    y = data_df["label"]
    over_sampler = RandomOverSampler(random_state=42)
    x_res, y_res = over_sampler.fit_resample(X.values.reshape(-1, 1), y)
    x_res = np.ravel(x_res)
    x_res = pd.Series(x_res)
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, stratify=y_res, test_size=0.25, random_state=245
    )
    vectorizer = TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test, y_train, y_test


@st.cache_resource
def confusion_matrix_func(model_name, _X_train, _X_test, _y_train, _y_test):
    """
    Generate a confusion matrix for a given model using the test data.
    """
    with st.spinner('Generating Confusion Matrix:'):
        time.sleep(1)
        model = model_name()
        model.fit(_X_train, _y_train)
        predicted = model.predict(_X_test)
        cm = confusion_matrix(_y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)

        # set axis labels and chart title
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

    st.success('Confusion Matrix generated!')

    # display the chart in Streamlit
    return st.pyplot(fig)


@st.cache_data
def reset_feature():
    """
    Rebuild features from the current dataset using preprocessing,
    oversampling, lemmatization, stop word removal, and TF-IDF vectorization.
    """
    df = load_dataset()
    X = df["text"]
    y = df["label"]
    over_sampler = RandomOverSampler(random_state=42)
    x_res, y_res = over_sampler.fit_resample(X.values.reshape(-1, 1), y)
    x_res = np.ravel(x_res)
    x_res = pd.Series(x_res)
    x_train, x_test, y_train, y_test = train_test_split(
        x_res, y_res, stratify=y_res, test_size=0.25, random_state=245
    )
    vectorizer = TfidfVectorizer(
        tokenizer=LemmaTokenizer(),
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    return x_train, x_test, y_train, y_test, vectorizer


@st.cache_resource
def load_transformer():
    """
    Load a pre-trained transformer model for sentiment analysis from the Hugging Face transformers library using the pipeline method.
    """
    sentiment_classifier = pipeline(
                                    model="distilbert-base-uncased-finetuned-sst-2-english",
                                    task="sentiment-analysis",
                                    top_k=None)
    return sentiment_classifier


def predict_pipeline(model, x_train, x_test, y_train, y_test):
    """
    Runs the given 'model' on the given training and testing data, and returns a dictionary containing classification
    report information, as well as the ROC AUC score for both the training and testing data (if the given model
    has a `predict_proba` method).
    """
    y_pred = model(x_test)
    classification_dict = classification_report(y_test, y_pred, output_dict=True)
    try:
        pr_train = model.predict_proba(x_train)[:, 1]
        pr_test = model.predict_proba(x_test)[:, 1]
        
        train_auc = roc_auc_score(y_train, pr_train)
        test_auc = roc_auc_score(y_test, pr_test)
    except AttributeError as e:
        train_auc = None
        test_auc = None
    return classification_dict, train_auc, test_auc


def predict_model(model_name, user_input, x_train, x_test, y_train, y_test, vectorizer, alpha=None, C=None, max_feat=None, n_estim=None, n_jobs=None, max_iterations=None, max_lr=None):
        """
        Train the selected model and predict for user_input.
        Returns:
        - data: list of (text, pred_int, pred_label_str, confidence_float)
        - result: classification report dict
        - train_auc, test_auc: AUCs if available else None
        """
        # Initialize model with stronger defaults
        if model_name == MultinomialNB:
            model = model_name(alpha=alpha or 1.0)
        elif model_name == LogisticRegression:
            model = model_name(
                C=C or 1.0,
                max_iter=max_lr or 1000,
                n_jobs=n_jobs,
                class_weight='balanced',
                solver='saga'
            )
        elif model_name == LinearSVC:
            model = model_name(
                C=C or 1.0,
                max_iter=max_iterations or 2000
            )
        else:
            model = model_name(n_estimators=n_estim or 100, max_features=max_feat or None)

        # Train
        model.fit(x_train, y_train)

        # Evaluate
        result, train_auc, test_auc = model_evaluate(model, x_train, x_test, y_train, y_test)

        # Predict for user inputs
        test = pd.Series(user_input)
        pred_test = vectorizer.transform(test)
        predictions = model.predict(pred_test)
        pred_to_label = {0: 'Negative', 1: 'Positive'}

        # Confidence calculation
        confidences = []
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(pred_test)
            # Confidence is max class probability
            confidences = (np.max(proba, axis=1) * 100.0).tolist()
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(pred_test)
            # Map distance to [50, 99] monotonically
            conf = 50.0 + 50.0 * (np.abs(df_vals) / (np.abs(df_vals) + 1.0))
            confidences = conf.tolist()
        else:
            confidences = [75.0] * len(predictions)

        # Make a list of user_text with sentiment and confidence
        data = []
        for t, pred, conf in zip(user_input, predictions, confidences):
            data.append((t, pred, pred_to_label[pred], float(np.round(conf, 2))))

        return data, result, train_auc, test_auc