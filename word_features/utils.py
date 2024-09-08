# Import libraries
import re
import nltk
from nltk.tokenize import word_tokenize
import syllapy
from spellchecker import SpellChecker




'''
Remove all charcters which are not alphabets or numbers
'''
def remove_non_alphanumeric(input_string):
    cleaned_string = re.sub(r'[^a-zA-Z0-9 ]', '', input_string)
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    return cleaned_string.strip()


'''
Filter all the word which are spelled incorrectly
'''
def filter_correct_words(word_list):
    spell = SpellChecker()
    correct_words = []
    for word in word_list:
        if word in spell:
            correct_words.append(word)
    return correct_words


'''
Calculate density of syllables when a list of words is given.
'''
def count_syllables(word_list):
    syllable = [0] * 6
    for word in word_list:
        syllables = syllapy.count(word)
        if syllables == 1:
            syllable[0] += 1
        elif syllables == 2:
            syllable[1] += 1
        elif syllables == 3:
            syllable[2] += 1
        elif syllables == 4:
            syllable[3] += 1
        elif syllables == 5:
            syllable[4] += 1
        else:
            syllable[5] += 1
    for i in range(6):
        syllable[i] /= len(word_list)
    return syllable


'''
Calculate density of parts of speech when a list of words is given.
'''
def count_pos(word_list):
    pos_descriptions = {
        'CC': 0,
        'CD': 0,
        'DT': 0,
        'EX': 0,
        'FW': 0,
        'IN': 0,
        'JJ': 0,
        'JJR': 0,
        'JJS': 0,
        'LS': 0,
        'MD': 0,
        'NN': 0,
        'NNS': 0,
        'NNP': 0,
        'NNPS': 0,
        'PDT': 0,
        'POS': 0,
        'PRP': 0,
        'PRP$': 0,
        'RB': 0,
        'RBR': 0,
        'RBS': 0,
        'RP': 0,
        'SYM': 0,
        'TO': 0,
        'UH': 0,
        'VB': 0,
        'VBD': 0,
        'VBG': 0,
        'VBN': 0,
        'VBP': 0,
        'VBZ': 0,
        'WDT': 0,
        'WP': 0,
        'WP$': 0,
        'WRB': 0
    }
    pos_tags = nltk.pos_tag(word_list)
    for tag in pos_tags:
        try:
            pos_descriptions[tag[1]] += 1
        except Exception as err:
            pass
    for k, v in pos_descriptions.items():
        pos_descriptions[k] = v / len(word_list)
    return pos_descriptions


'''
Count frequency of numbers when a string is given.
'''
def count_numbers(text):
    numbers = re.findall(r'\b\d+\b', text)
    return len(numbers)


'''
Extract features from text based on statistics of words
'''
def extract_word_features(text: str):
    features = {}
    text = remove_non_alphanumeric(text)
    words = word_tokenize(text)
    correct_words = filter_correct_words(words)
    features['correct_words'] = len(correct_words) / len(words)
    features['unique_words'] = len(set(correct_words)) / len(words)
    features['vocabulary_strength'] = len(correct_words) / 35000
    temp = count_syllables(correct_words)
    for i in range(len(temp)):
        features[f'syllable_{i+1}'] = temp[i]
    del temp
    features['numbers'] = count_numbers(text) / len(words)
    temp = count_pos(correct_words)
    features.update(temp)
    return features