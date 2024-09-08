# Import libraries
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import textstat
from tqdm import tqdm



'''
Extract features from text based on statistical property of sentences
'''
def extract_sentence_features(text: str):
    features = {}
    sentences = sent_tokenize(text)
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    features['smog_index'] = textstat.smog_index(text)
    features['coleman_liau_index'] = textstat.coleman_liau_index(text)
    features['automated_readability_index'] = textstat.automated_readability_index(text)
    features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    features['difficult_words'] = textstat.difficult_words(text)
    features['linsear_write_formula'] = textstat.linsear_write_formula(text)
    features['gunning_fog'] = textstat.gunning_fog(text)
    features['text_standard'] = textstat.text_standard(text, float_output=True)
    features['spache_readability'] = textstat.spache_readability(text)
    features['mcalpine_eflaw'] = textstat.mcalpine_eflaw(text)
    features['reading_time'] = textstat.reading_time(text)
    features['syllable_count'] = textstat.syllable_count(text)
    features['lexicon_count'] = textstat.lexicon_count(text)
    features['monosyllabcount'] = textstat.monosyllabcount(text)
    return features


'''
Function to extract features and save it
'''
def extract_and_save_features(row, id_col, input_col, folder_path):
    essay_id = row[id_col]
    text = row[input_col]
    features = extract_sentence_features(text)
    np.save(os.path.join(folder_path, str(essay_id) + '.npy'), np.array(features.values()))


'''
Given a dataframe extract features for all records
'''
def get_sentence_features(df: pd.DataFrame, id_col: str, input_col: str):
    folder_path = os.path.join('features', 'sentence_features')
    os.makedirs(folder_path, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda row: extract_and_save_features(row, id_col, input_col, folder_path), 
                               df.to_dict(orient='records')), 
                  desc='Extracting Sentence Features', total=df.shape[0]))