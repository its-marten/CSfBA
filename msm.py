from strsimpy.qgram import QGram
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import math
import re
from strsimpy.normalized_levenshtein import NormalizedLevenshtein as norm_lv
import textdistance as txtdist
import ast
from sklearn.cluster import AgglomerativeClustering
import hashlib
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from itertools import combinations
from lsh_marten import *

def get_index_by_title(title, df):
    index = df[df['title'] == title].index
    if index.empty:
        raise ValueError(f"Title '{title}' not found in the DataFrame.")
    return index[0]
    
def get_brand_name_by_idx(idx, df):
    feauture_map = get_fmap_by_idx(idx, df)
    brand = get_brand_name_by_fmap(feauture_map)  
    return brand

def get_fmap_by_idx(idx, df):
    features_product = df.iloc[idx]['featuresMap']
    return features_product

def get_brand_name_by_fmap(fmap):
    return fmap.get("brand") or fmap.get("brand name")

def get_shop_by_idx(idx, df):
    if idx is None or idx < 0 or idx >= len(df):
        raise ValueError(f"Invalid index: {idx}. DataFrame has {len(df)} rows.")
    product_shop = df.iloc[idx]["shop"]
    return product_shop

def get_title_by_idx(idx, df):
    p_title = df.iloc[idx]["title"]
    return p_title

##########################################################################################################################################
# Question here we check the max distance, rather than the distance of the 2 which one is better?
def calculate_qgram_similarity_version(string1, string2):
    qgram = QGram(3)
    n1 = len((string1))
    n2 = len((string2))
    value_sim = (n1 + n2 - qgram.distance(string1, string2))/ (n1 + n2)
    return value_sim

###################################################################################################
def split_mw(mw):
    # Use regular expressions to split the model word into numeric and non-numeric parts
    numeric_parts = re.findall(r'\d+\.?\d*', mw)
    non_numeric_parts = re.sub(r'\d+\.?\d*', '', mw)

    # Join numeric parts to form the complete numeric part
    numeric_part = ''.join(numeric_parts)

    return numeric_part, non_numeric_parts


def calculate_cosine_sim(a: str, b: str):
    set_a = set(a.split()) # ---> splits the title model words on the basis of a space, and puts them in a set
    set_b = set(b.split())
    # a and b are sets of elements
    intersection = set_a.intersection(set_b)
    intersection_size = len(intersection)
    return intersection_size / (math.sqrt(len(a)) * math.sqrt(len(b)))

def get_title_mws_by_index(product_index: int, df) -> set: # --> By product index
    return df.iloc[product_index]["title"]

def lv_sim(word_i, word_j) -> float:
    lv_dist = txtdist.levenshtein.normalized_distance(word_i,word_j)
    lv_sim = (1 - lv_dist)
    return lv_sim


def avg_lv_sim(X: set, Y: set) -> float:
    total_sim = 0
    total_length = 0
    for x in X:
        for y in Y:
            total_sim += lv_sim(x, y) * (len(x) + len(y))
            total_length += (len(x) + len(y))
    
    return total_sim / total_length

def calc_sim(string1, string2):
    qgram = QGram(3)
    n1 = len((string1))
    n2 = len((string2))
    value_sim = (n1 + n2 - qgram.distance(string1, string2))/ (n1 + n2)
    return value_sim

def same_shop(idx_i, idx_j, df):
    shop_i = get_shop_by_idx(idx_i, df)
    shop_j = get_shop_by_idx(idx_j, df)
    return shop_i == shop_j

def diff_brand(idx_i, idx_j, df):
    brand_i = get_brand_name_by_idx(idx_i, df)
    brand_j = get_brand_name_by_idx(idx_j, df)
    return brand_i == brand_j

def key(q: tuple) -> str:
    return q[0]

def value(q: tuple) -> str:
    return q[1]


def mw(C: set, D: set) -> float:
    # WIth C & D the set of matching model words
    intersection = C.intersection(D)
    union = C.union(D)
    return len(intersection) / len(union)

def text_distance(non_num_x, non_num_y, threshold: float) -> bool:
    dist = txtdist.levenshtein.normalized_distance(non_num_x, non_num_y)
    sim = 1 - dist
    #print(f"the sim for the non_num parts is {sim}")
    if sim >= threshold:
        #print(f"the non num parts are similar enough for threshold {threshold}")
            return True
    else:
        return False
    

def compute_values_sim(pi_idx, pj_idx, df, gamma):
    mk_sim = 0
    total_sim = 0
    matches = 0
    total_weight = 0

    # Print the indices and gamma value
    #print(f"Computing similarity for indices {pi_idx} and {pj_idx} with gamma {gamma}")

    mws_i = df.iloc[pi_idx]["model_words"].split()
    mws_j = df.iloc[pj_idx]["model_words"].split()

    # Print the model words
    #print(f"Model words for index {pi_idx}: {mws_i}")
    #print(f"Model words for index {pj_idx}: {mws_j}")

    kvp_i = get_fmap_by_idx(pi_idx, df)
    kvp_j = get_fmap_by_idx(pj_idx, df)

    # Print the key-value pairs
    #print(f"Key-value pairs for index {pi_idx}: {kvp_i}")
    #print(f"Key-value pairs for index {pj_idx}: {kvp_j}")

    nmk_i = list(kvp_i.keys())
    nmk_j = list(kvp_j.keys())

    for key_i in kvp_i.keys():
        for key_j in kvp_j.keys():  # Corrected typo here from kvp_j.jeys() to kvp_j.keys()
            key_sim = calculate_qgram_similarity_version(key_i, key_j)
            #print(f"Key similarity between {key_i} and {key_j}: {key_sim}")
            if key_sim > gamma:
                value_sim = calculate_qgram_similarity_version(kvp_i[key_i], kvp_j[key_j])
                #print(f"Value similarity between {kvp_i[key_i]} and {kvp_j[key_j]}: {value_sim}")

                total_sim += key_sim * value_sim
                matches += 1
                total_weight += key_sim

                #print(f"Total similarity: {total_sim}, Matches: {matches}, Total weight: {total_weight}")

                if key_i in nmk_i:
                    nmk_i.remove(key_i)
                if key_j in nmk_j:
                    nmk_j.remove(key_j)

    if total_weight > 0:
        mk_sim = total_sim / total_weight
    #print(f"Final mk_sim: {mk_sim}")

    nm_mw_i = set()
    for key in nmk_i:
        nm_mw_i.update(ex_mw(kvp_i[key], mws_i))
    #print(f"Non-matching model words for index {pi_idx}: {nm_mw_i}")

    nm_mw_j = set()
    for key in nmk_j:
        nm_mw_j.update(ex_mw(kvp_j[key], mws_j))
    #print(f"Non-matching model words for index {pj_idx}: {nm_mw_j}")

    if nm_mw_i or nm_mw_j:
        nmk_sim = mw(nm_mw_i, nm_mw_j)
    else:
        nmk_sim = 0
    #print(f"Final nmk_sim: {nmk_sim}")

    return mk_sim, nmk_sim, matches

def ex_mw(string, mws):
    extracted_mws = set()
    for mw in mws:
        if mw in string:
            extracted_mws.add(mw)
    return extracted_mws

def TMWA(alpha, beta, delta, eps, pi_idx, pj_idx, df):
    # Step 1: Calculate initial cosine similarity
    a = df.iloc[pi_idx]["title"]
    b = df.iloc[pj_idx]["title"]
    title_cos_sim = calculate_cosine_sim(a, b)
    #print(f"Initial cosine similarity: {title_cos_sim}")
    if title_cos_sim > alpha:
        print(f"Cosine similarity {title_cos_sim} > alpha {alpha}. Returning 1.")
        return 1

    # Step 2: Extract model words
    title_mws_i = get_title_mws_by_index(pi_idx, df).split()
    title_mws_j = get_title_mws_by_index(pj_idx, df).split()
    #print(f"Model words for index {pi_idx}: {title_mws_i}")
    #print(f"Model words for index {pj_idx}: {title_mws_j}")

    # Step checking whether they can be identified as different
    for mw_i in title_mws_i:
        for mw_j in title_mws_j:
            num_i, non_num_i = split_mw(mw_i)
            num_j, non_num_j = split_mw(mw_j)
            #print(f"Comparing {mw_i} and {mw_j}")
            #print(f"Non-numeric parts: {non_num_i} and {non_num_j}")
            #print(f"Numeric parts: {num_i} and {num_j}")

            if text_distance(non_num_i, non_num_j, 1-eps) and num_i != num_j:
                #print(f"Identified as different. Returning -1.")
                return -1

    # Step 3: Calculate initial similarity based on model words
    final_title_sim = beta * title_cos_sim + (1 - beta) * avg_lv_sim(title_mws_i, title_mws_j)
    #print(f"Initial similarity based on model words: {final_title_sim}")

    # Step 4: Iterate over model words to refine similarity
    found_valid_pair = False  # Track if any valid pair is found
    numerator = 0
    denominator = 0
    for mw_i in title_mws_i:
        for mw_j in title_mws_j:
            num_i, non_num_i = split_mw(mw_i)
            num_j, non_num_j = split_mw(mw_j)
            if text_distance(non_num_i, non_num_j, alpha) and num_i == num_j:
                found_valid_pair = True
                #print(f"Found valid pair: {mw_i} and {mw_j}")
                numerator += lv_sim(mw_i, mw_j) * (len(mw_i) + len(mw_j))
                denominator += (len(mw_i) + len(mw_j))
                #print(f"Numerator: {numerator}, Denominator: {denominator}")

    if found_valid_pair:
        final_title_sim = delta * (numerator / denominator) + (1 - delta) * final_title_sim
        #print(f"Refined final title similarity: {final_title_sim}")

    return final_title_sim

    
def get_min_features(idx_i, idx_j, df):
    # Get the feature maps (dicts) for both products
    features_i = get_fmap_by_idx(idx_i, df)
    features_j = get_fmap_by_idx(idx_j, df)
    
    # Calculate the number of features for each
    num_features_i = len(features_i)
    num_features_j = len(features_j)
    
    # Return the minimum
    return min(num_features_i, num_features_j)

def extract_values_to_set(features: dict) -> set:
    value_set = set()
    for value in features.values():
        # Convert value to string if it's not already
        value_str = str(value)
        value_set.add(value_str)
    return value_set  

def h_clustering(dissimilarity_matrix, eps, candidate_pairs_set):
    # Apply hierarchical clustering
    model = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=eps, metric = 'precomputed')
    cluster_labels = model.fit_predict(dissimilarity_matrix)

    #print(f"Threshold {eps}: Cluster labels assigned: {set(cluster_labels)}")

    # Group indices by cluster labels
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        cluster_dict[label].append(idx)

    print(f"Threshold {eps}: Cluster sizes: {[len(indices) for indices in cluster_dict.values()]}")

    # Generate valid pairs within candidate pairs only
    pairs = []
    for indices in cluster_dict.values():
        if len(indices) > 1:  # Only process clusters with more than one element
            cluster_pairs = combinations(indices, 2)
            pairs.extend(p for p in cluster_pairs if tuple(sorted(p)) in candidate_pairs_set)
            # Debug: Check filtering
            #print(f"Cluster indices: {indices}")

    return pairs

def MSM(df, cand_pairs, alpha, beta, gamma, eps_tmwa, mu, delta):
    n = df.shape[0]
    dist_matrix = np.full((n, n), 1e10)
    #print(f"Initialized distance matrix with size {n}x{n}")

    for pair in cand_pairs:
        pi_idx, pj_idx = pair  # Unpack the tuple into i and j
        #print(f"Processing pair: ({pi_idx}, {pj_idx})")

        shop_name_i = df.iloc[pi_idx]['shop']
        shop_name_j = df.iloc[pj_idx]['shop']
        brand_name_i = df.iloc[pi_idx]['brand']
        brand_name_j = df.iloc[pj_idx]['brand']

        #print(f"Shop names: {shop_name_i} and {shop_name_j}")
        #print(f"Brand names: {brand_name_i} and {brand_name_j}")

        if shop_name_i != shop_name_j and brand_name_i == brand_name_j:
            #print("Brand names match and shop names do not match. Proceeding with similarity calculations.")

            mk_sim, nmk_sim, num_matches = compute_values_sim(pi_idx, pj_idx, df, gamma)
            #print(f"compute_values_sim results - mk_sim: {mk_sim}, nmk_sim: {nmk_sim}, num_matches: {num_matches}")

            tmwm_sim = TMWA(alpha, beta, delta, eps_tmwa, pi_idx, pj_idx, df)
            #print(f"TMWA similarity: {tmwm_sim}")

            min_feat = get_min_features(pi_idx, pj_idx, df)
            #print(f"Minimum features: {min_feat}, the length of {pi_idx} is {df.iloc[pi_idx]["featuresMap"]} and the length of {pj_idx} is {df.iloc[pj_idx]["featuresMap"]}")

            if min_feat != 0 and tmwm_sim != 1:
                theta_1 = (1 - mu) * (num_matches / min_feat)
                theta_2 = 1 - mu - theta_1
                final_sim = theta_1 * mk_sim + theta_2 * nmk_sim + mu * tmwm_sim
            elif min_feat != 0 and tmwm_sim == 1:
                theta_1 = num_matches / min_feat
                theta_2 = 1 - theta_1
                final_sim = theta_1 * mk_sim + theta_2 * nmk_sim
            else:
                final_sim = mk_sim


            #print(f"Final similarity: {final_sim}")

            dist_matrix[pi_idx][pj_idx] = 1 - final_sim
            dist_matrix[pj_idx][pi_idx] = 1 - final_sim
            #print(f"Updated distance matrix for pair ({pi_idx}, {pj_idx}) and ({pj_idx}, {pi_idx})")

    return dist_matrix


def F1(precision, recall):
    if precision + recall == 0:
        print(f"recall is {recall} and precision = {precision}")
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

def tuning_cluster(dissimilarity_matrix, truePairs, candidate_pairs, tuning_parameters):
    
    # Precompute all candidate pairs as a set for quick lookups
    candidate_pairs_set = set(map(tuple, candidate_pairs))

    savelist = {}
    f1_scores = []
    for para in tuning_parameters:
        print(f"Evaluating threshold: {para}")
        pairs = h_clustering(dissimilarity_matrix, para, candidate_pairs_set)

        # Debug: Check generated pairs
        #print(f"Threshold {para}: Pairs generated: {len(pairs)}")
        #print(f"Sample pairs: {pairs[:5]}")  # Check a few pairs
        
        # Calculate TP, TN, FP, FN
        set_candidate_pairs = set(map(tuple, pairs))
        set_true_duplicates = set(map(tuple, truePairs))
        npair = len(pairs)
        if npair == 0:
            continue
        all_pairs = set(combinations(range(dissimilarity_matrix.shape[0]), 2))
        non_duplicate_pairs = all_pairs - set_true_duplicates
        TP = len(set_candidate_pairs.intersection(set_true_duplicates))
        FP = len(set_candidate_pairs - set_true_duplicates)
        FN = len(set_true_duplicates - set_candidate_pairs)

        # Debug: Metrics
        print(f"Threshold {para}: TP = {TP}, FP = {FP}, FN = {FN}")
        
        if TP + FP == 0 or TP + FN == 0:
            print(f"Threshold {para}: Skipped due to no positive predictions or true pairs.")
            continue

        precision_PQ = TP / (TP + FP)
        recall_PC = TP / (TP + FN)
        F1_value = F1(precision_PQ, recall_PC)

        print(f"Threshold {para}: Precision = {precision_PQ}, Recall = {recall_PC}, F1 = {F1_value}")

        fraction_comp = len(candidate_pairs) / len(all_pairs)
        savelist[F1_value] = [precision_PQ, recall_PC, fraction_comp, para, npair]
        f1_scores.append(F1_value)

    if not f1_scores:
        #print("No valid F1 scores computed.")
        return {}

    f1_score = max(f1_scores)
    rest = savelist[f1_score]

    return {
        'pair_quality(precision)': rest[0],
        'pair_completeness(recall)': rest[1],
        'F1': f1_score,
        'fraction_comp': rest[2],
        'threshold': rest[3],
        'numbPair': rest[4],
    }
