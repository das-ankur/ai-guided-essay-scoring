# Import libraries
import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm



'''
Extract features from text based on linguistic properties
'''
def extract_linguistic_features(text):
    doc = nlp(text)
    features = {"GPE": 0, "PERCENT": 0, "NORP": 0, "ORG": 0, "CARDINAL": 0, "MONEY": 0, "DATE": 0, 
                "LOC": 0, "PERSON": 0, "QUANTITY": 0, "EVENT": 0, "ORDINAL": 0, "WORK_OF_ART": 0, 
                "LAW": 0, "PRODUCT": 0, "TIME": 0, "FAC": 0, "LANGUAGE": 0}
    for entity in doc.ents:
        if entity.label_ in features:
            features[entity.label_] += 1
    tenses = [i.morph.get("Tense") for i in doc]
    tenses = [i[0] for i in tenses if i]
    tense_counts = Counter(tenses)
    features['past_tense_ratio'] = tense_counts.get("Past", 0) / (tense_counts.get("Pres", 0) + tense_counts.get("Past", 0) + 1e-5)
    features['present_tense_ratio'] = tense_counts.get("Pres", 0) / (tense_counts.get("Pres", 0) + tense_counts.get("Past", 0) + 1e-5)
    paragraph = text.split('\n\n')
    features['paragraph_count'] = len(paragraph)
    features['avg_chars_by_paragraph'] = np.mean([len(paragraph) for paragraph in paragraph])
    features['avg_words_by_paragraph'] = np.mean([len(nltk.word_tokenize(paragraph)) for paragraph in paragraph])
    features['avg_sentences_by_paragraph'] = np.mean([len(nltk.sent_tokenize(paragraph)) for paragraph in paragraph]) 
    analyzer = SentimentIntensityAnalyzer()
    sentences = nltk.sent_tokenize(text)
    compound_scores, negative_scores, positive_scores, neutral_scores = [], [], [], []
    for sentence in sentences:
        scores = analyzer.polarity_scores(sentence)
        compound_scores.append(scores['compound'])
        negative_scores.append(scores['neg'])
        positive_scores.append(scores['pos'])
        neutral_scores.append(scores['neu'])
    features["mean_compound"] = np.mean(compound_scores)
    features["mean_negative"] = np.mean(negative_scores)
    features["mean_positive"] = np.mean(positive_scores)
    features["mean_neutral"] = np.mean(neutral_scores)
    features["std_compound"] = np.std(compound_scores)
    features["std_negative"] = np.std(negative_scores)
    features["std_positive"] = np.std(positive_scores)
    features["std_neutral"] = np.std(neutral_scores)
    return features


'''
Function to extract features and save it
'''
def extract_and_save_features(row, id_col, input_col, folder_path):
    essay_id = row[id_col]
    text = row[input_col]
    features = extract_linguistic_features(text)
    np.save(os.path.join(folder_path, str(essay_id) + '.npy'), np.array(features.values()))


'''
Given a dataframe extract features for all records
'''
def get_sentence_features(df: pd.DataFrame, id_col: str, input_col: str):
    folder_path = os.path.join('features', 'linguistic_features')
    os.makedirs(folder_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda row: extract_and_save_features(row, id_col, input_col, folder_path), 
                               df.to_dict(orient='records')), 
                  desc='Extracting Linguistic Features', total=df.shape[0]))
