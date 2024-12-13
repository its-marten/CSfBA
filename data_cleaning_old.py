import pandas as pd
import re

def drop_url(data):
    
    data = data.drop(['url'], axis=1)
    
    return data

def clean_title_old(data):
    
    data['title'] = data['title'].str.lower() # Standardize everything to lower case

    # Define the regex patterns for cleaning
    title_patterns = {
        r'[\'"”]': 'inch' , r'\s*inches?\b': 'inch',  # Standardize inch
        r'[\/\(\)\-\[\]\:\–\—\,\&\+\|]':' ',  # Remove special characters
        r'(?<!\d)\.(?!\d|\w)': ' ', # Remove dot unless it is between 2 numbers
        r'hertz': 'hz', r'\b-\s*hz\b|\s*hz': 'hz',  # Standardize hertz
        r'\b(\w+)\b(?=.*\b\1\b)' : r'', # Remove duplicate alphanumeric strings
        r'\s+': ' ',  # Remove multiple spaces
        r'^\s+|\s+$': ''  # Remove leading and trailing spaces
    }

    for title_pattern, replacement in title_patterns.items():
        data['title'] = data['title'].str.replace(title_pattern, replacement, regex=True)
    
    return data


def get_brand(data):
    
    def pop_brand(features):
        if isinstance(features, dict) and 'Brand' in features:
            return features.pop('Brand').lower()
        return None

    data['brand'] = data['featuresMap'].apply(lambda x: pop_brand(x))

    return data


def get_model_words_old(data):

    model_word_title_pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^\d, ]+[0-9]+))[a-zA-Z0-9]*)'

    data['model_words_title'] = data['title'].apply(lambda x: ' '.join([i[0] for i in re.findall(model_word_title_pattern, x.lower())]))

    model_word_feature_pattern = r'(\d+(\.\d+)?[a-zA-Z]+$|\d+(\.\d+)?$)'

    def extract_words_from_features(features):
        model_words = []
        
        if isinstance(features, dict):  # Ensure featuresMap is a dictionary
            for key, value in features.items():
                if isinstance(value, str):
                    # Extract words matching the pattern
                    model_words += re.findall(model_word_feature_pattern, value.lower())
                elif isinstance(value, (int, float)): 
                    # Handle numeric values as strings
                    model_words += re.findall(model_word_feature_pattern, str(value))
        
        # Join the extracted words and return them as a single string
        return ' '.join([word[0] for word in model_words])

    # Apply the extraction logic to the featuresMap column
    data['model_words_features'] = data['featuresMap'].apply(lambda x: extract_words_from_features(x))

    # Get all model words
    data['model_words'] = data['model_words_title'] + ' ' + data['model_words_features']

    # Remove duplicate words
    def remove_duplicates_from_model_words(row):
        if isinstance(row, str):
            words = row.split()
            unique_words = set(words)
            return ' '.join(sorted(unique_words))
        else:
            return ''

    data['model_words'] = data['model_words'].apply(lambda x: remove_duplicates_from_model_words(str(x)))
    
    return data

def clean_data_old(data):
    
    data = drop_url(data)
    data = clean_title_old(data)
    data = get_brand(data)
    data = get_model_words_old(data)
    
    return data
