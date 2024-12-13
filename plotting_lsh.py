# %%
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
from data_cleaning import *
from data_cleaning_old import *
from msm import *
from lsh import *

# Open JSON file
file_name = 'TVs-all-merged.json'
data = open_json(file_name)

# Clean Data
df_lsh, df_msm, df_cleaning, df = clean_full_data(data)
old_data = clean_data_old(data)

def get_bootstrap_samples(data, ratio, number_bootstrap, seed):
    bootstrap = []  
    test = []  

    np.random.seed(seed)

    for i in range(number_bootstrap):
        sample = data.sample(frac=ratio, replace=True, random_state=i)  
        out_of_sample_data = data.loc[data.index.difference(sample.index)]  
        bootstrap.append(sample.reset_index(drop=True))  
        test.append(out_of_sample_data.reset_index(drop=True))  
    
    return bootstrap, test


def bootstrap_samples_lsh(data, ratio, number_bootstrap, b_r_comb, seed):
    results_lsh = []
    bootstrap, _ = get_bootstrap_samples(data,ratio, number_bootstrap,seed)  

    # Dictionary to accumulate results for each combination of bands and rows
    accumulated_results = {comb: [] for comb in b_r_comb}

    for i, sample in enumerate(bootstrap):
        print(f"Currently on bootstrap {i + 1}")

        true_pairs = get_true_pairs(sample)
        print(f"Number of True Pairs: {len(true_pairs)}")

        binary_matrix = generate_binary_matrix(sample)

        for comb in b_r_comb:
            bands, rows = comb
            signature_matrix = generate_signature_matrix(binary_matrix, bands, rows)
            candidate_pairs = lsh(signature_matrix, bands)
            print(f"Number of Candidate Pairs for bands {bands} and rows{rows}: {len(candidate_pairs)}")

            # Compute evaluation metrics
            tp = len(candidate_pairs.intersection(true_pairs))  # True positives
            fp = len(candidate_pairs - true_pairs)             # False positives
            fn = len(true_pairs - candidate_pairs)             # False negatives

            pq = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            pc = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
            f1_star = (2 * pq * pc) / (pq + pc) if (pq + pc) > 0 else 0
            fraction_comparisons = len(candidate_pairs) / ((len(sample) * (len(sample) - 1)) / 2)

            t = (1 / bands) ** (1 / rows)

            # Store results
            accumulated_results[comb].append({
                'bands': bands,
                'rows': rows,
                't': t,
                'f1_star': f1_star,
                'pc': pc,
                'pq': pq,
                'fraction_comparisons': fraction_comparisons
            })

    # Calculate averages for each combination of bands and rows
    for comb, results in accumulated_results.items():
        avg_f1_star = sum(r['f1_star'] for r in results) / len(results)
        avg_pc = sum(r['pc'] for r in results) / len(results)
        avg_pq = sum(r['pq'] for r in results) / len(results)
        avg_fraction_comparisons = sum(r['fraction_comparisons'] for r in results) / len(results)
        bands, rows = comb

        results_lsh.append({
            'bands': bands,
            'rows': rows,
            't': (1 / bands) ** (1 / rows),
            'f1_star': avg_f1_star,
            'pc': avg_pc,
            'pq': avg_pq,
            'fraction_comparisons': avg_fraction_comparisons
        })

    return results_lsh
    

def plot_results(results, results_old):
    
    # Convert results to a DataFrame for easier plotting
    df = pd.DataFrame(results)
    df_old = pd.DataFrame(results_old)

    # Sort the DataFrame by fraction_comparisons for proper line plotting
    df = df.sort_values(by='fraction_comparisons')
    df_old = df_old.sort_values(by='fraction_comparisons')

    # Plot PC vs Fraction Comparisons
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(df['fraction_comparisons'], df['pc'], c='blue', label='PC New')
    plt.plot(df_old['fraction_comparisons'], df_old['pc'], c='red', label='PC Old')
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('Pair Completeness (PC)')
    plt.title('PC vs Fraction Comparisons')
    plt.legend()

    # Plot PQ vs Fraction Comparisons
    plt.subplot(1, 3, 2)
    plt.plot(df['fraction_comparisons'], df['pq'], c='blue', label='PQ New')
    plt.plot(df_old['fraction_comparisons'], df_old['pq'], c='red', label='PQ Old')
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('Pair Quality (PQ)')
    plt.title('PQ vs Fraction Comparisons')
    plt.legend()

    # Plot F1* vs Fraction Comparisons
    plt.subplot(1, 3, 3)
    plt.plot(df['fraction_comparisons'], df['f1_star'], c='blue', label='F1* New')
    plt.plot(df_old['fraction_comparisons'], df_old['f1_star'], c='red', label='F1* Old')
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('F1* Measure')
    plt.title('F1* vs Fraction Comparisons')
    plt.legend()

    plt.tight_layout()
    plt.show()


number_bootstrap = 5
ratio = 0.63
seed=42

band_rows_combinations = [(500, 1), (250, 2), (125, 4), (100, 5), (50, 10), (25,20), (20, 25), (10, 50), (5, 100), (4, 125), (2, 250), (1, 500)]

lsh_results = bootstrap_samples_lsh(df_lsh, ratio, number_bootstrap, band_rows_combinations,seed)
lsh_results_old = bootstrap_samples_lsh(old_data, ratio, number_bootstrap, band_rows_combinations,seed)

print(lsh_results, lsh_results_old)

### LSH PLOTS ###
plot_results(lsh_results, lsh_results_old)

# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results, results_old):
    # Convert results to a DataFrame for easier plotting
    df = pd.DataFrame(results)
    df_old = pd.DataFrame(results_old)

    # Sort the DataFrame by fraction_comparisons for proper line plotting
    df = df.sort_values(by='fraction_comparisons')
    df_old = df_old.sort_values(by='fraction_comparisons')

    # Plot PC vs Fraction Comparisons
    plt.figure(figsize=(6, 4))
    plt.plot(df['fraction_comparisons'], df['pc'], c='black', label='MSMJ', linewidth=1.5)
    plt.plot(df_old['fraction_comparisons'], df_old['pc'], c='black', linestyle='--', label='MSMP+', linewidth=1.5)
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('Pair Completeness (PC)')
    plt.title('PC vs Fraction Comparisons')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot PQ vs Fraction Comparisons
    plt.figure(figsize=(6, 4))
    plt.plot(df['fraction_comparisons'], df['pq'], c='black', label='MSMJ', linewidth=1.5)
    plt.plot(df_old['fraction_comparisons'], df_old['pq'], c='black', linestyle='--', label='MSMP+', linewidth=1.5)
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('Pair Quality (PQ)')
    plt.title('PQ vs Fraction Comparisons')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot F1* vs Fraction Comparisons
    plt.figure(figsize=(6, 4))
    plt.plot(df['fraction_comparisons'], df['f1_star'], c='black', label='MSMJ', linewidth=1.5)
    plt.plot(df_old['fraction_comparisons'], df_old['f1_star'], c='black', linestyle='--', label='MSMP+', linewidth=1.5)
    plt.xlabel('Fraction Comparisons')
    plt.ylabel('F1* Measure')
    plt.title('F1* vs Fraction Comparisons')
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
plot_results(lsh_results, lsh_results_old)

# %%



