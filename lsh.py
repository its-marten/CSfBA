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

def generate_binary_matrix(data):
    unique_words = list(set(word for row in data['model_words'] for word in row.split()))
    binary_matrix = pd.DataFrame(0, columns=unique_words, index=data.index)
    
    for unique_word in unique_words:
        binary_matrix[unique_word] = data['model_words'].apply(lambda x: 1 if unique_word in x else 0)
    
    binary_matrix = binary_matrix.transpose()
    binary_matrix = binary_matrix.to_numpy()
    
    return binary_matrix

def generate_signature_matrix(binary_data, bands, rows, seed = 42):
    permutations = bands * rows  
    words, signatures = binary_data.shape
    
    sig_matrix = np.full((permutations, signatures), np.iinfo(np.int32).max, dtype=np.int32) # Initialize signature matrix 
    
    def generate_permutation_indices(words):
        sequence = np.arange(1, words + 1)
        np.random.shuffle(sequence)
        return sequence
    
    for i in range(permutations): 
        signature = np.full((1, signatures), np.inf)
        permutation = generate_permutation_indices(words)
        
        for row in range(words):  
            nonzero_indices = np.where(binary_data[row, :] == 1)[0]
            if len(nonzero_indices) == 0:
                continue
            for col in nonzero_indices: 
                if signature[0, col] > permutation[row]:
                    signature[0, col] = permutation[row]
                    
        sig_matrix[i, :] = signature.astype(int)

    return sig_matrix

def lsh(signature_matrix, bands):

    signature_matrix = pd.DataFrame(signature_matrix)
    candidates = []

    rows_per_band = len(signature_matrix) // bands
    for band_index in range(bands):
        start_row = band_index * rows_per_band
        end_row = start_row + rows_per_band if band_index < bands - 1 else len(signature_matrix)
        band = signature_matrix.iloc[start_row:end_row]

        hash_buckets = {}

        for column in band.columns:
            hashed_values = ''.join(str(int(x)) for x in band[column])
            bucket_hash = int(hashlib.sha256(hashed_values.encode('utf-8')).hexdigest(), 16)
            if bucket_hash not in hash_buckets:
                hash_buckets[bucket_hash] = []
            hash_buckets[bucket_hash].append(column)
    
        for bucket_columns in hash_buckets.values():
            for i in range(len(bucket_columns) - 1):
                for j in range(i + 1, len(bucket_columns)):
                    candidates.append((bucket_columns[i], bucket_columns[j]))

    candidate_pairs = set(candidates)

    return candidate_pairs

def get_true_pairs(data):
    true_duplicates = defaultdict(list)
    for idx, product in data.iterrows():
        model_id = product['modelID']
        true_duplicates[model_id].append(idx)
    
    true_pairs = set()
    for indices in true_duplicates.values():
        if len(indices) > 1:
            for pair in combinations(indices, 2):
                true_pairs.add(tuple(sorted(pair)))
    
    return true_pairs

def test_lsh_combinations(binary_matrix, true_pairs, band_row_combinations, seed=42):
    results = []
    
    for bands, rows in band_row_combinations:
        print(f"Testing with bands={bands}, rows={rows}")
        t = (1/bands)**(1/rows)
        signature_matrix = generate_signature_matrix(binary_matrix, bands=bands, rows=rows)
        
        candidate_pairs = lsh(signature_matrix, bands)
        
        candidate_pairs = set(candidate_pairs)
        print(f"Sample Candidate Pairs: {list(candidate_pairs)[:5]}")
        print(f"Sample True Pairs: {list(true_pairs)[:5]}")
        
        tp = len(candidate_pairs.intersection(true_pairs))
        fp = len(candidate_pairs - true_pairs)           
        fn = len(true_pairs - candidate_pairs)            

        pq = tp / (tp + fp) if tp + fp > 0 else 0 
        pc = tp / (tp + fn) if tp + fn > 0 else 0 
        f1_star = (2 * pq * pc) / (pq + pc) if (pq + pc) > 0 else 0 

        missing_pairs = true_pairs - candidate_pairs
        print(f"Missing Pairs: {len(missing_pairs)}")
        if missing_pairs:
            print(f"Sample Missing Pairs: {list(missing_pairs)[:5]}")

        results.append({
            "bands": bands,
            "rows": rows,
            "pair_quality": pq,
            "pair_completeness": pc,
            "f1_star": f1_star,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "candidate_pairs": len(candidate_pairs),
            "missing_pairs": len(missing_pairs),
            "threshold t": t
        })
    
   
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_star", ascending=False).reset_index(drop=True)


    print("\nBest combinations based on F1* Measure:")
    print(results_df.head(15))

    return results_df

def generate_band_row_combinations(matrix_shape, tolerance=0.05):
    rounded_rows = int(round(matrix_shape / 100.0) * 100)
    target = rounded_rows // 2  
    
    combinations = []
    for bands in range(2, int(math.sqrt(target)) + 1):  
        rows = target // bands  
        if rows * bands == target: 
            combinations.append((bands, rows))
            if bands != rows:
                combinations.append((rows, bands))
    return combinations

def lsh_performance(true_pairs, candidate_pairs):
    print(f"Sample Candidate Pairs: {list(candidate_pairs)[:5]}")
    print(f"Sample True Pairs: {list(true_pairs)[:5]}")

   
    tp = len(candidate_pairs.intersection(true_pairs)) 
    fp = len(candidate_pairs - true_pairs)
    fn = len(true_pairs - candidate_pairs)             

    pq = tp / (tp + fp) if tp + fp > 0 else 0  
    pc = tp / (tp + fn) if tp + fn > 0 else 0  
    f1_star = (2 * pq * pc) / (pq + pc) if (pq + pc) > 0 else 0 

    print(f"Pair Quality (PQ): {pq:.4f}")
    print(f"Pair Completeness (PC): {pc:.4f}")
    print(f"F1* Measure: {f1_star:.4f}")

    missing_pairs = true_pairs - candidate_pairs
    print(f"Missing Pairs: {len(missing_pairs)}")
    if missing_pairs:
        print(f"Sample Missing Pairs: {list(missing_pairs)[:5]}")

    return pq, pc, f1_star, tp
