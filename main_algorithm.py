import numpy as np  
import pandas as pd  
import hashlib
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from itertools import combinations
from msm import *

def generate_binary_matrix(data):
    # Extract unique words from model words
    unique_words = list(set(word for row in data['model_words'] for word in row.split()))
    binary_matrix = pd.DataFrame(0, columns=unique_words, index=data.index)
    
    # Fill the binary matrix
    for unique_word in unique_words:
        binary_matrix[unique_word] = data['model_words'].apply(lambda x: 1 if unique_word in x else 0)
    
    binary_matrix = binary_matrix.transpose()
    binary_matrix = binary_matrix.to_numpy()
    
    return binary_matrix

def generate_signature_matrix(binary_data, bands, rows, seed = 42):
    permutations = bands * rows  # Number of permutations
    words, signatures = binary_data.shape
    
    sig_matrix = np.full((permutations, signatures), np.iinfo(np.int32).max, dtype=np.int32) # Initialize signature matrix 

    # Set the random seed
    np.random.seed(seed)
    
    # Generate random permutations for MinHashing
    def generate_permutation_indices(words):
        sequence = np.arange(1, words + 1)
        np.random.shuffle(sequence)
        return sequence
    
    for i in range(permutations):  # For each permutation
        signature = np.full((1, signatures), np.inf)
        permutation = generate_permutation_indices(words)
        
        for row in range(words):  # For each row in binary data
            nonzero_indices = np.where(binary_data[row, :] == 1)[0]
            if len(nonzero_indices) == 0:
                continue
            for col in nonzero_indices:  # Update the signature if the permuted value is smaller
                if signature[0, col] > permutation[row]:
                    signature[0, col] = permutation[row]
                    
        sig_matrix[i, :] = signature.astype(int)

    return sig_matrix

def generate_candidate_pairs(signature_matrix, bands, seed=42):

    signature_matrix = pd.DataFrame(signature_matrix)
    candidates = []

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Split the signature matrix into bands
    rows_per_band = len(signature_matrix) // bands
    for band_index in range(bands):
        start_row = band_index * rows_per_band
        end_row = start_row + rows_per_band if band_index < bands - 1 else len(signature_matrix)
        band = signature_matrix.iloc[start_row:end_row]

        hash_buckets = {}

        for column in band.columns:
            # Create a hashable string representation of the column
            hashed_values = ''.join(str(int(x)) for x in band[column])

            # Use hashlib to generate a deterministic hash
            bucket_hash = int(hashlib.sha256(hashed_values.encode('utf-8')).hexdigest(), 16)

            # Add columns with the same hash to the same bucket
            if bucket_hash not in hash_buckets:
                hash_buckets[bucket_hash] = []
            hash_buckets[bucket_hash].append(column)

        # Generate candidate pairs from the buckets
        for bucket_columns in hash_buckets.values():
            for i in range(len(bucket_columns) - 1):
                for j in range(i + 1, len(bucket_columns)):
                    candidates.append((bucket_columns[i], bucket_columns[j]))

    # Remove duplicate candidate pairs
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

# def F1(precision, recall):
#     if precision + recall == 0:
#         return 0  
#     return (2 * precision * recall) / (precision + recall)

# def h_clustering(dissimilarity_matrix, eps, candidate_pairs_set):
    
#     # Apply hierarchical clustering
#     model = AgglomerativeClustering(n_clusters=None, linkage='average', distance_threshold=eps, metric = 'precomputed')
#     cluster_labels = model.fit_predict(dissimilarity_matrix)

#     # Group indices by cluster labels
#     cluster_dict = defaultdict(list)
#     for idx, label in enumerate(cluster_labels):
#         cluster_dict[label].append(idx)

#     # Generate valid pairs within candidate pairs only
#     pairs = []
#     for indices in cluster_dict.values():
#         if len(indices) > 1:  # Only process clusters with more than one element
#             cluster_pairs = combinations(indices, 2)
#             pairs.extend(p for p in cluster_pairs if tuple(sorted(p)) in candidate_pairs_set)

#     return pairs

# def TMWA(alpha, beta, delta, eps, a: str, b: str, df):
    
#     # Step 1: Calculate initial cosine similarity
#     name_cos_sim = calculate_cosine_sim(a, b)
#     if name_cos_sim > alpha:
#         return 1  # High similarity, return early

#     # Step 2: Extract model words
#     mws_a = get_title_mws_by_title(a, df)
#     mws_b = get_title_mws_by_title(b, df)

#     # Step 3: Calculate initial similarity based on model words
#     final_name_sim = beta * name_cos_sim + (1 - beta) * avgLvSim(mws_a, mws_b)
#     #print(f"{final_name_sim} is the initial avglvsim")

#     # Step 4: Iterate over model words to refine similarity
#     found_valid_pair = False  # Track if any valid pair is found
#     for mw_a in mws_a:
#         num_a, non_num_a = split_mw(mw_a)
#         for mw_b in mws_b:
#             num_b, non_num_b = split_mw(mw_b)
           
#             # Step 4.1: Check text distance between model words
#             if text_distance(num_a, non_num_a, num_b, non_num_b, threshold=0.7):
#                 found_valid_pair = True
#                 mw_sim_val = avgLvSimMW(mws_a, mws_b, threshold=0.7)

#                 # Update final similarity using delta weight
#                 final_name_sim = delta * mw_sim_val + (1 - delta) * final_name_sim
#             else: 
                
#                 continue
   
#     # Step 5: Final check based on threshold
#     if found_valid_pair and final_name_sim > eps:
#         return final_name_sim
#     else:
#         return -1  # No valid similarity or below threshold


# def MSM(df, cand_pairs, alpha, beta, gamma, eps_cluster, eps_tmwa, mu, delta, needed_output):
#     n = df.shape[0]
#     dist = np.full((n, n), 1e10)

#     for pair in cand_pairs:
#         pi_idx, pj_idx = pair  # Unpack the tuple into i and j
        
#         p_i = get_title_by_idx(pi_idx, df)
#         p_j = get_title_by_idx(pj_idx, df)
        
#         shop_name_i = df.iloc[pi_idx]['shop']
#         shop_name_j = df.iloc[pj_idx]['shop']

#         brand_name_i = df.iloc[pi_idx]['brand']
#         brand_name_j = df.iloc[pj_idx]['brand']
        
#         if brand_name_i != brand_name_j or shop_name_i == shop_name_j:
 
#             continue
#         else:
#             keys_to_remove_i = set()
#             keys_to_remove_j = set()
#             sim = 0
#             avg_sim = 0
#             m = 0
#             w = 0
#             nmk_i, nmk_j = find_non_matching_kvps(pi_idx, pj_idx, df)
  
#             for key_i, val_i in nmk_i.items():
#                 for key_j, val_j in nmk_j.items():
#                     keySim = calc_sim(key_i, key_j)
#                     if keySim > gamma: 
#                         valueSim = calc_sim(val_i, val_j)
#                         weight = keySim
#                         sim = sim + weight * valueSim
#                         m = m + 1
#                         w = w + weight  
#             if w > 0:
#                 avg_sim = sim / w

#             mw_perc = mw(ex_mw(nmk_i), ex_mw(nmk_j))
#             title_sim = TMWA(alpha, beta, delta, eps_tmwa, p_i, p_j, df)
#             min_features = get_min_features(pi_idx, pj_idx, df)

#             if min_features == 0:
#                 h_sim = mu * title_sim
              
#             else:
               
#                 if title_sim == -1:
#                     theta1 = m/min_features
#                     theta2 = 1 - theta1
#                     h_sim = theta1 * avg_sim + theta2 * mw_perc
#                     (f"enough features, sim is {h_sim}, but title sim is -1")
#                 else:
#                     theta1 = (1 - mu) * (m / min_features)
#                     theta2 = 1 - mu - theta1

#                     h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * title_sim

#             dist[pi_idx][pj_idx] = 1 - h_sim
#             dist[pj_idx][pi_idx] = 1 - h_sim
          
#     if needed_output == "dist_matrix":
#         return dist
#     else:
#         return h_clustering(dist, eps_cluster)


# def MSM(df, cand_pairs, alpha, beta, gamma, eps_cluster, eps_tmwa, mu, delta, needed_output):
#     n = df.shape[0]
#     dist = np.full((n, n), 1e10)
#   #  print(f"Initialized distance matrix with size {n}x{n}")

#     for pair in cand_pairs:
#         pi_idx, pj_idx = pair  # Unpack the tuple into i and j
        
#         p_i = get_title_by_idx(pi_idx, df)
#         p_j = get_title_by_idx(pj_idx, df)
        
        
#         shop_name_i = df.iloc[pi_idx]['shop']
#         shop_name_j = df.iloc[pj_idx]['shop']

#         brand_name_i = df.iloc[pi_idx]['brand']
#         brand_name_j = df.iloc[pj_idx]['brand']
        
#         if brand_name_i != brand_name_j or shop_name_i == shop_name_j:
#             #print(f"Skipping pair due to same shop or different brand: {brand_name_i}, {brand_name_j} - Shop names: {shop_name_i}, {shop_name_j} ")
#             continue
#         else:
#             #print(f"Proceeding with pair: {p_i}, {p_j} - Shop names: {shop_name_i}, {shop_name_j}")
#             keys_to_remove_i = set()
#             keys_to_remove_j = set()
#             sim = 0
#             avg_sim = 0
#             m = 0
#             w = 0
#             # Iterate through all feature keys of both items
#             all_keys_i = df.iloc[pi_idx]['featuresMap']
#             all_keys_j = df.iloc[pj_idx]['featuresMap']
#             # Print the number of keys before processing
#           #  print(f"Number of keys in product {pi_idx} feature map before processing: {len(all_keys_i)}")
# #            print(f"Number of keys in product {pj_idx} feature map before processing: {len(all_keys_j)}")

#             for key_i, val_i in all_keys_i.items():
#                 for key_j, val_j in all_keys_j.items():
#                     keySim = calc_sim(key_i, key_j)
#                     if keySim > gamma: 
#               #          print(f"the keysim is {keySim} and higher than 0.7")
#                         #print(f'ERRORRRRRR Keys: {key_i} and {key_j} are empty')
#                         valueSim = calc_sim(val_i, val_j)
#                         weight = keySim
#                         sim += weight * valueSim
#                         m += 1
#                         w += weight
                        
#                         # Mark these keys for removal after comparisons are complete
#                         keys_to_remove_i.add(key_i)
#                         keys_to_remove_j.add(key_j)

#             # Remove matching keys after the loop
#             for key in keys_to_remove_i:
#                 all_keys_i.pop(key, None)
#             for key in keys_to_remove_j:
#                 all_keys_j.pop(key, None)  
#             # Print the number of keys after processing
#          #   print(f"Number of keys in product {pi_idx} feature map after processing: {len(all_keys_i)}")
#        #     print(f"Number of keys in product {pj_idx} feature map after processing: {len(all_keys_j)}")

#             if w > 0:
#                 avg_sim = sim / w
#                 #print(f"Average similarity after matching: {avg_sim}")\

#             if len(ex_mw(all_keys_i)) == 0 and len(ex_mw(all_keys_j)) == 0:
#                 mw_perc = 0
#             else:
#                 mw_perc = mw(ex_mw(all_keys_i), ex_mw(all_keys_j))
#         #    print(f"function has made it till here mw_perc is {mw_perc}")
#             title_sim = TMWA(alpha, beta, delta, eps_tmwa, p_i, p_j, df)
#             min_features = get_min_features(pi_idx, pj_idx, df)
#         #    print(f"Title similarity (TMWA): {title_sim}")
#             #print(f"{min_features} is the min amount of features")
#             if min_features == 0:
#                 h_sim = mu * title_sim
#                 #(f"not enough features, sim is {h_sim}")
#             else:
#                 #print("enough features")
#                 if title_sim == -1:
#                     theta1 = m/min_features
#                     theta2 = 1 - theta1
#                     h_sim = theta1 * avg_sim + theta2 * mw_perc
#                     #(f"enough features, sim is {h_sim}, but title sim is -1")
#                 else:
#                     theta1 = (1 - mu) * (m / min_features)
#                     theta2 = 1 - mu - theta1
#                     #print(f"m = {m}, theta1 is {theta1}, theta2 is {theta2}, mu is {mu}")

#                     h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * title_sim
#                     #(f"enough features, sim is {h_sim}")
#         #    print(f"the h_sim is {h_sim}")
#             dist[pi_idx][pj_idx] = 1 - h_sim
#             dist[pj_idx][pi_idx] = 1 - h_sim
          
#     if needed_output == "dist_matrix":
#         #print(f"Returning distance matrix with shape: {dist.shape}")
#         return dist
#     else:
#         return h_clustering(dist, eps_cluster)


# def tuning_cluster(dissimilarity_matrix, true_pairs, candidate_pairs, tuning_parameters):
    
#     # Precompute all candidate pairs as a set for quick lookups
#     candidate_pairs_set = set(map(tuple, candidate_pairs))

#     savelist = {}
#     f1_scores = []
#     for para in tuning_parameters:
#         pairs_msm = h_clustering(dissimilarity_matrix, para, candidate_pairs_set)
        
#         # Calculate TP, TN, FP, FN
#         set_candidate_lsh = set(map(tuple, candidate_pairs))
#         set_true_duplicates = set(map(tuple, true_pairs))
#         possible_true_pairs = set_candidate_lsh.intersection(set_true_duplicates)

#         candidate_pairs_msm = set(map(tuple, pairs_msm))

        
#         npair = len(candidate_pairs_msm)
#         all_pairs = set(combinations(range(dissimilarity_matrix.shape[0]), 2))
#         non_duplicate_pairs = all_pairs - set_true_duplicates
        
#         TP = len(candidate_pairs_msm.intersection(possible_true_pairs))
#         FP = len(candidate_pairs_msm) - TP
#         FN = len(possible_true_pairs) - TP

#         precision_PQ = TP / (TP + FP) if (TP + FP) > 0 else 0
#         recall_PC = TP / (TP + FN) if (TP + FN) > 0 else 0
#         F1_value = F1(precision_PQ, recall_PC)

#         fraction_comp = len(candidate_pairs) / len(all_pairs)
#         savelist[F1_value] = [precision_PQ, recall_PC, fraction_comp, para, npair]
#         f1_scores.append(F1_value)

#     f1_score = max(f1_scores)
#     rest = savelist[f1_score]

#     return {
#         'precision': rest[0],
#         'recall': rest[1],
#         'F1': f1_score,
#         'fraction_comp': rest[2],
#         'threshold': rest[3],
#         'number of pairs': rest[4],
#     }
