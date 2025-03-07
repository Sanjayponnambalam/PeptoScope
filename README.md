# A Machine Learning Model Using Anti-Microbial Peptides in Diagnosing Cancer

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Objective](#objective)
4. [Problem Statement](#problem-statement)
5. [Challenges](#challenges)
6. [Literature Survey](#literature-survey)
7. [Requirement Analysis](#requirement-analysis)
8. [Implementation](#implementation)
9. [Results & Outputs](#results--outputs)
10. [Conclusion](#conclusion)
11. [References](#references)

## Introduction
Cancer is a global health concern, and timely diagnosis is crucial. Conventional diagnostic methods have limitations, leading to delayed detection and treatment. This project presents a novel approach leveraging machine learning techniques to utilize antimicrobial peptides (AMPs) in cancer diagnosis.

## Motivation
Anticancer peptides have shown promising therapeutic effects with minimal side effects. The primary motivation is to develop a machine learning-based model for cancer detection using AMPs, aiming for precise and non-invasive early diagnosis.

## Objective
- Investigate the feasibility of machine learning combined with AMPs for cancer diagnosis.
- Develop a robust predictive model for distinguishing cancerous from normal tissues.
- Improve early detection methods for better patient outcomes.

## Problem Statement
Current cancer diagnostic methods lack sensitivity for early detection. This project integrates AMPs and clinical data with machine learning to create an accurate, non-invasive diagnostic tool.

## Challenges
1. Availability of high-quality AMP sequence datasets.
2. Ensuring model generalization across diverse cancer types.
3. Handling cancer cell heterogeneity.
4. Interpreting complex machine learning models.
5. Selecting and engineering relevant AMP features.

## Literature Survey
Studies have shown that sequence analysis plays a vital role in bioinformatics, particularly in disease detection. Previous models had accuracy rates below 75%. This project aims to achieve over 90% accuracy, leveraging novel biomarkers and machine learning techniques.

## Requirement Analysis
1. **Data Requirements:** High-quality datasets from WHO and other sources.
2. **Software Requirements:**
   - Conda (for package management)
   - Pfeature (for computing peptide properties)
   - CD-HIT (for filtering redundant sequences)
   - Scikit-learn (for machine learning)

## Implementation
### Step 1: Install Conda
```bash
conda install -c conda-forge conda
```

### Step 2: Install Pfeature
```bash
pip install pfeature
```

### Step 3: Install CD-HIT
```bash
conda install -c bioconda cd-hit
```

### Step 4: Download and preprocess datasets
- Collect peptide sequences in FASTA format.
- Remove redundant sequences using CD-HIT.

### Step 5: Feature Extraction
- Use Pfeature to compute amino acid compositions and other properties.

### Step 6: Model Training
- Perform binary classification using scikit-learn.
- Split data into training and testing sets.

### Step 7: Train the Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
```

### Step 8: Evaluate Model
- Plot Receiver Operating Characteristic (ROC) curve.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

![ROC Curve](images/roc_curve.png)

## Results & Outputs
- **Model Performance:**
  - The model demonstrates high accuracy and reliability in distinguishing between cancerous and non-cancerous samples.
- **Receiver Operating Characteristic (ROC) Curve Analysis:**
  - The ROC curve illustrates the effectiveness of the model in classifying peptide sequences.
- **Feature Importance:** The most important peptide feature was identified as 'K'.

## Conclusion
This study successfully leveraged machine learning and antimicrobial peptides for cancer detection. The developed model offers a promising, non-invasive diagnostic approach with high accuracy. Future improvements include additional dataset integration and advanced deep learning techniques.

## References
1. Sharma, M. (2020). "A CNN-Based K-Mer Classification of Anti-Microbial Peptide Sequences" - IEEE.
2. Zhao, Z. (2019). "A computational model for anti-cancer drug sensitivity prediction" - BioCAS.
3. Gull, S. (2022). "AMP0: Species-Specific Prediction of Anti-microbial Peptides" - IEEE/ACM Transactions.

For more details, refer to the complete research report.

