import pandas as pd
import numpy as np
import re
import json
from collections import Counter
from sklearn.model_selection import train_test_split


# Opening Json File
def open_json(file_name):
    # Read JSON file
    with open(file_name, 'r') as file:
        content = json.load(file)

    # Flatten lists and convert to dataframe
    dataset = [item for sublist in content.values() for item in sublist]
    data = pd.DataFrame(dataset)
    return data



# Main dataframe 

# Dropping url column
def drop_url(data):
    
    data = data.drop(['url'], axis=1)
    
    return data

# TITLE 

# cleaning title format based on paper
def clean_title(data):


    # Standardize everything to lower case
    data['title'] = data['title'].str.lower()

    # Define the regex patterns for cleaning
    title_patterns = {
        r'[\'"”]': 'inch ',  # Standardize inch
        r'\s*inches?\b': 'inch',
        r'(\d+):(\d+)': r'\1by\2', # changing : to by between 2 numbers
        r'\b:|:\b': '', # deleting trailing or beginning colon
        r'[\/\(\)\-\[\]\–\—\,\&\+\|]': '',  # Remove special characters
        r'(?<!\d)\.(?!\d|\w)': ' ',  # Remove dot unless it is between 2 numbers
        r'hertz': 'hz', 
        r'\b-\s*hz\b|\s*hz': 'hz',  # Standardize hertz
        #r'\b(\w+)\b(?=.*\b\1\b)': '',  # Remove duplicate alphanumeric strings
        r'\b(\d+\.\d+inch)\b(?=.*\b\1\b)': '',
        r'^\s+|\s+$': '',  # Remove leading and trailing spaces
        r'newegg\.com|thenerds\.net| - best buy| - thenerds\.net': ''  # Remove website names
    }

    # Apply the patterns to clean the title
    for title_pattern, replacement in title_patterns.items():
        data['title'] = data['title'].str.replace(title_pattern, replacement, regex=True)
    
    # Explicitly remove extra spaces (double or more) with a single space
    data['title'] = data['title'].str.replace(r'\s+', ' ', regex=True)
    
    return data
# Format last parts of the title words like 46910 should be 46.9
def reformat_inch_fractions_df(df):

    def reformat_title(title):
        # Find all matches for numbers followed by "inch"
        matches = re.findall(r'(\d+)inch', title)
        updated_title = title  # Start with the original title
        for match in matches:
            number = match  # Numeric part
            if len(number) > 3:  # Ensure at least 4 digits
                whole_part = int(number[:2])  # First two digits as the whole part
                if len(number) > 4:  # Handle case where length is greater than 4
                    numerator = int(number[2])  # Third digit
                    denominator = int(number[3:])  # Last two digits combined
                    fractional_value = numerator / denominator  # Compute the fraction
                else:  # Standard case (length is exactly 4)
                    numerator = int(number[2])  # Third digit
                    denominator = int(number[3])  # Fourth digit
                    fractional_value = numerator / denominator  # Compute the fraction
                new_value = f"{whole_part + fractional_value:.1f}inch"
                # Replace the current match with the new value
                updated_title = updated_title.replace(f"{number}inch", new_value)
        return updated_title

    # Apply the reformat_title function to the 'title' column
    df['title'] = df['title'].apply(reformat_title)
    return df

# Function to clean the title column
def clean_title_column(df):
    def clean_title(title):
        # Remove "refurbished"
        title = title.replace("refurbished", "")
        # Round floats in "inch" to the nearest integer
        title = re.sub(r"(\d+\.\d+)(?=inch)", lambda x: str(round(float(x.group(1)))), title)
        # Remove duplicate words
        words = title.split()
        unique_words = []
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
        title = ' '.join(unique_words)
        return title

    if 'title' in df.columns:
        df['title'] = df['title'].apply(clean_title)
    else:
        raise ValueError("The DataFrame does not contain a 'title' column.")
    
    return df


# FEATURES
def clean_features_values(df):

    def process_fraction(value):
        def fraction_to_decimal(match):
            whole_number = match.group(1)
            numerator = int(match.group(2))
            denominator = int(match.group(3))
            fraction = numerator / denominator
            # Combine whole number and fraction without trailing ".0"
            return f"{int(whole_number) + fraction:.10f}".rstrip('0').rstrip('.')
        # Match the whole number, numerator, and denominator
        return re.sub(r'(\d+)-(\d+)/(\d+)', fraction_to_decimal, value)
    

    # Define the regex patterns and their replacements
    feature_patterns = [
        (r'(\d+\.\d+)"\s*x\s*(\d+\.\d+)"\s*x\s*(\d+\.\d+)"\s*\(.*?\)', r'\1x\2x\3'),  # Format dimensions and remove trailing WxHxD
        (r'[\'"”]', 'inch'),  # Standardize inch
        (r'[\/\(\)\-\[\]\–\—\,\&\+\#\|]', ''),  # Remove special characters
        (r'\s*inches?\b', 'inch'),
        (r'(\d+):(\d+)', r'\1by\2'), # changing : to by between 2 numbers
        (r'(?<!\d)\.(?!\d|\w)', ''),  # Remove dot unless between two numbers
        (r'\(\s*$', ' '),  # Remove standalone open parenthesis at end
        (r'^\s*\)', ''),  # Remove standalone close parenthesis at start
        (r'\s*x\s*', 'x'),  # Remove spaces around "x" in dimensions
        (r'\s*\+\s*', ''),  # Remove spaces and plus sign
        (r'\$', ''),  # Remove dollar signs
        (r'°', ''),  # Remove degree symbol
        (r'²', ''),  # Remove squared symbol
        (r'\blbs\.?\b', r'lb'),
        (r'(\d+(\.\d+)?)\s*(lb|lbs)\.*', r'\1lb'),
        (r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', ''),
        (r'(?<=[a-zA-Z])\[a-zA-Z]', lambda m: m.group(0)[0]),  # Remove slash between two letters 
    ]

    def process_features_map(features_map):
        updated_map = {}
        for key, value in features_map.items():
            # Clean the key
            for pattern, replacement in feature_patterns:
                key = re.sub(pattern, replacement, key)
            if isinstance(value, str):
                value = process_fraction(value)
            # Clean the value if it's a string
            if isinstance(value, str):
                for pattern, replacement in feature_patterns:
                    value = re.sub(pattern, replacement, value)

            # Update the cleaned key-value pair in the new dictionary
            updated_map[key] = value
        return updated_map
    
    df['featuresMap'] = df['featuresMap'].apply(process_features_map)
    return df

def lowercase_featuremap(df):    
    def process_features_map(features_map):
        cleaned_dict = {}
        for key, value in features_map.items():
            # Convert key and value to lowercase strings
            key_str = str(key).lower()
            value_str = str(value).lower()
            cleaned_dict[key_str] = value_str
        return cleaned_dict

    # Apply processing to each row's 'featuresMap'
    df['featuresMap'] = df['featuresMap'].apply(process_features_map)
    return df

def extract_brand_from_features(df):
    def get_brand(features_map):
        # Ensure the input is a dictionary
        if not isinstance(features_map, dict):
            return None
        
        # Look for 'brand' or 'brand name' in the featuresMap
        brand = features_map.get('brand')
        
        if brand:
            # Normalize the brand (lowercase and strip trailing colons)
            return brand.strip().lower().rstrip(':')
        
        # Return None if no brand is found
        return None
    
    # Apply the function to the 'featuresMap' column and create a new column 'brand'
    df['brand'] = df['featuresMap'].apply(get_brand)
    
    return df


# KEYS
def clean_dict_keys_df(df):
    chars_to_strip = ":/\\;()[]{}"
    
    def clean_keys(features):
        cleaned_dict = {}
        for key, value in features.items():
            # Strip unwanted leading and trailing characters from keys
            key_str = str(key).strip(chars_to_strip)
            cleaned_dict[key_str] = value
        return cleaned_dict
    
    # Apply the cleaning function to each dictionary in 'featuresMap'
    df['featuresMap'] = df['featuresMap'].apply(clean_keys)
    return df

# BRAND
def clean_brand_key(df, column_name="featuresMap"):
    def standardize_key(key):
        # Rename "brand name" and "brand name:" to "brand"
        if key.lower() in ["brand name", "brand name:"]:
            return "brand"
        return key  # Return key unchanged if no match

    # Apply standardization to all keys in the feature maps
    for idx, row in df.iterrows():
        if isinstance(row[column_name], dict):  # Ensure the row contains a dictionary
            standardized_map = {}
            for key, value in row[column_name].items():
                standardized_key = standardize_key(key)
                standardized_map[standardized_key] = value
            df.at[idx, column_name] = standardized_map

    return df

def extract_and_remove_brand(df, column_name="featuresMap"):
    def extract_brand(features_map):
        if isinstance(features_map, dict):
            # Extract the brand value if it exists and remove it from the dictionary
            return features_map.pop("brand", None)
        return None

    # Create a new column 'brand' with extracted brand values
    df["brand"] = df[column_name].apply(extract_brand)
    return df

def fill_brand_from_title(df, brand_column="brand", title_column="title"):
    def extract_brand_from_title(row):
        if pd.isna(row[brand_column]):  # Check if brand is None or NaN
            words = row[title_column].split()  # Split title into words
            if words:
                # Skip "refurbished" if it's the first word
                word = words[1] if words[0].lower() == "refurbished" else words[0]
                # Return the word if it's alphabetical (not alphanumeric)
                if word.isalpha():
                    return word
        return row[brand_column]  # Return existing brand if not missing

    df[brand_column] = df.apply(extract_brand_from_title, axis=1)
    return df

def clean_brand_names(df, brand_column="brand"):
    replacements = {"lg electronics": "lg", "jvc tv": "jvc"}
    df[brand_column] = df[brand_column].apply(
        lambda brand: replacements.get(brand.lower().strip(), brand) if isinstance(brand, str) else brand
    )
    return df

# SHOP
def remove_com_from_shop(df):
    # Remove ".com" from the 'shop' column
    df['shop'] = df['shop'].str.replace('.com', '', regex=False)
    df['shop'] = df['shop'].str.replace('.net', '', regex=False)
    return df

# SPLITTING LSH AND MSM DF BASED ON HIGH FREQ FEATURES
def get_high_frequency_features(features_count_df, percentage, total_count):
    # Calculate the count threshold based on the percentage
    threshold = total_count * (percentage / 100)

    # Filter features with counts above the threshold
    low_frequency_features = features_count_df[features_count_df['Count'] >= threshold]

    # Return the list of feature names
    return low_frequency_features["Key"].tolist()

def filter_features_in_featuremap(df, features_to_keep,  features_column = "featuresMap"):
    for idx, row in df.iterrows():
        features = row[features_column]
        if isinstance(features, dict):  # Ensure featuresMap is a dictionary
            # Retain only the keys present in features_to_keep
            filtered_features = {key: value for key, value in features.items() if key in features_to_keep}
            df.at[idx, features_column] = filtered_features  # Update the DataFrame entry

    return df

def unique_keys_count(df, column_name = "featuresMap"):
    # Initialize a Counter to store the counts of keys
    key_counter = Counter()

    # Iterate over each row in the column
    for row in df[column_name]:
        if isinstance(row, dict):  # Ensure the row is a dictionary
            key_counter.update(row.keys())

    # Convert to a DataFrame for better visualization (optional)
    key_counts_df = pd.DataFrame(key_counter.items(), columns=["Key", "Count"]).sort_values(by="Count", ascending=False)

    return key_counts_df

# MODEL WORDS
def model_words_title(data):


    model_word_title_pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^\d, ]+[0-9]+))[a-zA-Z0-9]*)'

    data['model_words_title'] = data['title'].apply(lambda x: ' '.join([i[0] for i in re.findall(model_word_title_pattern, x.lower())]))

    return data

def model_words_features(data, feature_column="featuresMap", new_column="model_words_features"):
    def extract_from_features(features):
        extracted_values = []
        if isinstance(features, dict):  # Ensure it's a dictionary
            for key, value in features.items():
                # Use an enhanced regex to ensure floats and their units (e.g., "13.5inch") are captured correctly
                matches = re.findall(
                    r'\b(?:[0-9]+(?:\.[0-9]+)?(?:inch|mm|cm|hz|hz|px)?|[0-9]+x[0-9]+|[0-9]+by[0-9]+|[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+)\b',
                    str(value)
                )
                extracted_values.extend(matches)
        return ' '.join(extracted_values)

    # Apply the extraction function row by row
    data[new_column] = data[feature_column].apply(lambda x: extract_from_features(x))

    return data

# Function to remove duplicates in the model_words column
def clean_model_words_column(df):
    if 'model_words' in df.columns:
        df['model_words'] = df['model_words'].apply(lambda x: ' '.join(dict.fromkeys(x.split())))
    else:
        raise ValueError("The DataFrame does not contain a 'model_words' column.")
    
    return df


# DATA CLEANING ALGORYTHM


def clean_full_data(df):
    # First making a copy of the dataset to do all the cleaning
    df_cleaning =  df.copy()
    df_cleaning = drop_url(df_cleaning)

    # Cleaning the title
    df_cleaning = clean_title(df_cleaning)
    df_cleaning = reformat_inch_fractions_df(df_cleaning)
    df_cleaning = clean_title_column(df_cleaning)

    # cleaning the features
    df_cleaning = lowercase_featuremap(df_cleaning)
    df_cleaning = clean_features_values(df_cleaning)

    # cleaning keys
    df_cleaning = clean_dict_keys_df(df_cleaning)

    # cleaning the brand key and column
    df_cleaning = clean_brand_key(df_cleaning)
    df_cleaning = extract_and_remove_brand(df_cleaning)
    df_cleaning = fill_brand_from_title(df_cleaning)
    df_cleaning = clean_brand_names(df_cleaning)

    # cleaning shop column
    df_cleaning = remove_com_from_shop(df_cleaning)

    # model words for title
    df_cleaning = model_words_title(df_cleaning)
    count_unqiue_features_df = unique_keys_count(df_cleaning)

    high_freq_features = get_high_frequency_features(count_unqiue_features_df, 50 ,df_cleaning.shape[0])


    # setting up dataframe for msm
    df_msm =  df_cleaning.copy()
    df_msm = model_words_features(df_msm)
    df_msm['model_words'] = df_msm['model_words_title'] + ' ' + df_msm['model_words_features']
    df_msm = clean_model_words_column(df_msm)

    # setting up dataframe for lsh
    df_lsh = df_cleaning.copy()
    df_lsh = filter_features_in_featuremap(df_lsh, high_freq_features)
    df_lsh = model_words_features(df_lsh)
    df_lsh['model_words'] =  df_lsh['model_words_title'] + ' ' + df_lsh['model_words_features']
    df_lsh = clean_model_words_column(df_lsh)


    return df_lsh, df_msm, df_cleaning, df