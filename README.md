This repository contains the code for the individual assignment of **Computer Science for Business Analytics**. The code infrastructure was co-authored with **Sara Kaczmarek**, with differences mainly in the data cleaning approaches. The algorithms were developed collaboratively, following a shared thought process.

---

## Overview

The repository includes scripts for data cleaning, data cleaning for the old dataset, Locality Sensitive Hashing (LSH), and the Multi-component Similarity Method (MSM). These scripts work together to process the dataset, run the algorithms, and generate the plots used in the final report.

---

## Components

### **Data Cleaning**
- The `data_cleaning.py` file contains a function that automates all necessary data cleaning for the given JSON dataset. This function ensures the dataset is preprocessed correctly for subsequent steps.
- The `data_cleaning_old.py` file contains the data cleaning methods used in the MSMP+ paper, allowing for comparison of results between the new and old cleaning approaches.

---

### **LSH Algorithm**
- The `LSH_plotting.py` file runs 5 bootstraps on the dataset, testing different combinations of rows and bands (defined as variables in the script). 
- By importing the data cleaning files, you can obtain the necessary dataframes for LSH, including one for the new cleaning approach and one for the old cleaning approach.
- The resulting plots illustrate the algorithm's performance and are used in the report.
- Supporting functions for the LSH algorithm are implemented in `LSH.py`.

---

### **MSM Algorithm**
- The `MSM_plotting.py` file runs the MSM algorithm using pre-defined functions.
- By importing the data cleaning files, you can obtain the necessary dataframes for MSM, including one for the new cleaning approach and one for the old cleaning approach.
- This script produces the F1 measure plots, including the final plot featured in the report.

---

## Usage

### **Data Cleaning**
- Use `data_cleaning.py` to preprocess your dataset. Import the cleaning function into your script or run it directly on your JSON file.

---

### **Run LSH**
- Execute `LSH_plotting.py` to perform the LSH algorithm. Adjust rows and bands as needed in the script. This will generate plots for analysis.

---

### **Run MSM**
- Run `MSM_plotting.py` to execute the MSM algorithm and generate the F1 measure plot.

---
