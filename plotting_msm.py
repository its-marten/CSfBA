# %%
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from data_cleaning import *
from data_cleaning_old import *
from msm import *
#from evaluation import *

# Open JSON file
file_name = 'TVs-all-merged.json'
data = open_json(file_name)

# Clean Data
df_lsh, df_msm, df_cleaning, df = clean_full_data(data)
old_data = clean_data_old(data)

def bootstrap_samples(data1, data2, ratio, number_bootstrap, seed):
    if not data1.index.equals(data2.index):
        raise ValueError("The indices of data1 and data2 must be identical.")

    bootstrap_data1 = []
    bootstrap_data2 = []
    test_data1 = []
    test_data2 = []

    np.random.seed(seed)  # Set the seed for reproducibility
    indices = data1.index.to_numpy()  # Convert indices to a NumPy array
    total_length = len(indices)
    sample_size = int(ratio * total_length)

    for i in range(number_bootstrap):
        # Choose a random starting index
        start_idx = np.random.randint(0, total_length)
        
        # Generate sampled indices by wrapping around if necessary
        sampled_indices = np.concatenate((
            indices[start_idx:],
            indices[:start_idx]
        ))[:sample_size]

        # Find out-of-sample indices
        out_of_sample_indices = np.setdiff1d(indices, sampled_indices)

        # Subset both dataframes based on sampled and out-of-sample indices
        bootstrap_data1.append(data1.loc[sampled_indices].reset_index(drop=True))
        bootstrap_data2.append(data2.loc[sampled_indices].reset_index(drop=True))
        test_data1.append(data1.loc[out_of_sample_indices].reset_index(drop=True))
        test_data2.append(data2.loc[out_of_sample_indices].reset_index(drop=True))

    return bootstrap_data1, bootstrap_data2, test_data1, test_data2



def get_bootstrap_samples_old(data, ratio, number_bootstrap, seed):
    bootstrap = []  
    test = []  

    np.random.seed(seed)

    for i in range(number_bootstrap):
        sample = data.sample(frac=ratio, replace=True, random_state=i)  
        out_of_sample_data = data.loc[data.index.difference(sample.index)]  
        bootstrap.append(sample.reset_index(drop=True))  
        test.append(out_of_sample_data.reset_index(drop=True))  
    
    return bootstrap, test
    

def bootstrap_samples_msm(df_msm, df_lsh, ratio, number_bootstrap, b_r_comb, tuning_parameters, alpha, beta, gamma, mu, delta, epsilon_TMWM, seed):
    results_msm = []

    # Generate synchronized bootstrap samples for df_msm and df_lsh
    bootstrap_msm, bootstrap_lsh, _, _ = bootstrap_samples(df_msm, df_lsh, ratio, number_bootstrap, seed)

    # Dictionary to accumulate results for each combination of bands and rows
    accumulated_results = {comb: [] for comb in b_r_comb}

    for i, (msm_sample, lsh_sample) in enumerate(zip(bootstrap_msm, bootstrap_lsh)):
        print(f"Currently on bootstrap {i + 1}")
        print(len(lsh_sample), len(msm_sample))

        # Use LSH bootstrap for binary matrix
        true_pairs = get_true_pairs(lsh_sample)
        print(f"Number of True Pairs: {len(true_pairs)}")

        binary_matrix = generate_binary_matrix(lsh_sample)

        for comb in b_r_comb:
            bands, rows = comb

            # Create candidate pairs using LSH for the given bands and rows
            signature_matrix = generate_signature_matrix(binary_matrix, bands, rows)
            candidate_pairs = lsh(signature_matrix, bands)

            print(f"Number of Candidate Pairs for bands {bands} and rows {rows}: {len(candidate_pairs)}")

            # Use MSM bootstrap for dissimilarity matrix
            dissimilarity_matrix = MSM(msm_sample, candidate_pairs, alpha, beta, gamma, epsilon_TMWM, mu, delta)

            # Tune clusters and evaluate
            result = tuning_cluster(dissimilarity_matrix, true_pairs, candidate_pairs, tuning_parameters)

            # Store results for this bootstrap and combination of bands and rows
            accumulated_results[comb].append({
                'bootstrap': i + 1,
                'bands': bands,
                'rows': rows,
                **result
            })

    # Aggregate results into accumulated_results
    for comb, results in accumulated_results.items():
        avg_f1 = sum(r['F1'] for r in results) / len(results)
        avg_fraction_comparisons = sum(r['fraction_comp'] for r in results) / len(results)
        bands, rows = comb

        # Store aggregated results
        results_msm.append({
            'bands': bands,
            'rows': rows,
            'F1': avg_f1,
            'fraction_comp': avg_fraction_comparisons
        })

    return results_msm


def bootstrap_samples_old(data, ratio, number_bootstrap, b_r_comb, tuning_parameters, alpha, beta, gamma, mu, delta, epsilon_TMWM, seed):
    results_msm = []

    # Generate synchronized bootstrap samples for df_msm and df_lsh
    bootstrap, _ = get_bootstrap_samples_old(data, ratio, number_bootstrap, seed)

    # Dictionary to accumulate results for each combination of bands and rows
    accumulated_results = {comb: [] for comb in b_r_comb}

    for i, sample in enumerate(bootstrap):
        print(f"Currently on bootstrap {i + 1}")

        # Use LSH bootstrap for binary matrix
        true_pairs = get_true_pairs(sample)
        print(f"Number of True Pairs: {len(true_pairs)}")

        binary_matrix = generate_binary_matrix(sample)

        for comb in b_r_comb:
            bands, rows = comb

            # Create candidate pairs using LSH for the given bands and rows
            signature_matrix = generate_signature_matrix(binary_matrix, bands, rows)
            candidate_pairs = lsh(signature_matrix, bands)

            print(f"Number of Candidate Pairs for bands {bands} and rows {rows}: {len(candidate_pairs)}")

            # Use MSM bootstrap for dissimilarity matrix
            dissimilarity_matrix = MSM(sample, candidate_pairs, alpha, beta, gamma, epsilon_TMWM, mu, delta)

            # Tune clusters and evaluate
            result = tuning_cluster(dissimilarity_matrix, true_pairs, candidate_pairs, tuning_parameters)

            # Store results for this bootstrap and combination of bands and rows
            accumulated_results[comb].append({
                'bootstrap': i + 1,
                'bands': bands,
                'rows': rows,
                **result
            })

    # Aggregate results into accumulated_results
    for comb, results in accumulated_results.items():
        avg_f1 = sum(r['F1'] for r in results) / len(results)
        avg_fraction_comparisons = sum(r['fraction_comp'] for r in results) / len(results)
        bands, rows = comb

        # Store aggregated results
        results_msm.append({
            'bands': bands,
            'rows': rows,
            'F1': avg_f1,
            'fraction_comp': avg_fraction_comparisons
        })

    return results_msm


def plot_results(results):
    # Convert results to a DataFrame for easier plotting
    df = pd.DataFrame(results)

    # Sort the DataFrame by fraction_comparisons for proper line plotting
    df = df.sort_values(by='fraction_comp')

    # Plot F1 vs Fraction Comparisons
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)

    # Plot the F1 measure with a thinner line
    plt.plot(df['fraction_comp'], df['F1'], c='black', linewidth=1, label='F1 New')

    # Add a dashed vertical line at 0.49
    plt.axhline(y=0.49, color='gray', linestyle='--', linewidth=1)

    # Labels, title, and legend
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('F1 Measure')
    plt.title('F1 vs Fraction Comparisons')
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


tuning_parameters = [0.1, 0.3, 0.5, 0.75, 0.9] 
gamma, alpha, beta, mu, delta, epsilon_TMWM = 0.7, 0.5, 0.2, 0.5, 0.6, 0
number_bootstrap = 3
ratio = 0.63
seed=42

band_rows_combinations = [(500,1), (250, 2), (125, 4), (100, 5), (50, 10), (25,20), (20, 25), (10, 50), (5, 100), (4, 125), (2, 250), (1, 500)]
#band_rows_combinations = [(100, 1), (50, 2), (25, 4), (20, 5), (10, 10), (5, 20), (4,25), (2, 50), (1, 100)] # testing for only half of the observations

msm_results = bootstrap_samples_msm(df_msm, df_lsh, ratio, number_bootstrap, band_rows_combinations, tuning_parameters, alpha, beta, gamma, mu, delta, epsilon_TMWM, seed)

### MSM PLOTS ###
#plot_results(msm_results, msm_results_old)

# %%
plot_results(msm_results)


# %%



