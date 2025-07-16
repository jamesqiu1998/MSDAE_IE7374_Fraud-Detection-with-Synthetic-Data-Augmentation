# MSDAE_IE7374_Fraud-Detection-with-Synthetic-Data-Augmentation

This project aims to improve credit card fraud detection by addressing the extreme class imbalance in fraud datasets. Using the publicly available [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, we train a Variational Autoencoder (VAE) on fraudulent transactions to generate realistic synthetic fraud data. This augmented dataset is then used to train a classification model, such as a Random Forest, to improve detection performance.

---

## 📚 Final Topic Overview

Fraudulent transactions represent a tiny fraction of the dataset (~0.17%). Training models on such imbalanced data leads to poor generalization on minority (fraud) classes. We use a VAE trained solely on fraud samples to generate synthetic fraudulent data, augment the original dataset, and build a more balanced training set for a downstream classifier.

---

## 📦 Dataset Description

- **Total Transactions**: 284,806  
- **Fraudulent Transactions**: 492  
- **File Size**: ~150 MB  
- **Format**: CSV (Tabular)  
- **Source**: Collected in 2013 by Worldline and the ULB Machine Learning Group from European cardholders.  
- **Features**:
  - Time, Amount
  - 28 anonymized PCA components (V1–V28)

The original dataset is extremely imbalanced, which justifies the need for data augmentation using generative models.

---

## 🔍 Model Selection: Variational Autoencoder (VAE)

We selected the **Variational Autoencoder** because:

- It works well with **structured tabular data**, unlike many generative models designed for images or text.
- It's more stable and easier to train on **small or imbalanced datasets** compared to GANs.
- It learns general patterns from data, providing a **privacy-friendly** way to synthesize realistic fraud records.
- Synthetic data helps mitigate class imbalance, giving the classifier **more fraud examples** to learn from.

### 🔧 VAE Architecture

The VAE consists of:

- **Encoder**: Learns to represent the input (fraud samples) as a distribution by outputting a mean and standard deviation vector.
- **Decoder**: Reconstructs fraud-like samples by drawing from this latent distribution.

By learning the distribution of fraudulent transactions, the VAE is capable of generating synthetic data that mimics real fraud behavior.

---

## 🧠 Project Workflow

1. **Train VAE** on the 492 fraud samples.
2. **Generate synthetic fraud data** from the trained VAE.
3. **Combine** real and synthetic fraud data with the original dataset to reduce class imbalance.
4. **Train a classifier** (e.g., Random Forest) on the augmented dataset.
5. **Evaluate** the model on the original dataset using metrics such as precision, recall, and F1-score.

---

## 📈 Expected Outcomes

- A trained VAE capable of producing realistic fraudulent transaction data.
- An augmented dataset with improved class balance.
- A classification model that performs more consistently and accurately on fraud detection tasks.

---

## ⚙️ Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection-vae-augmentation.git
   cd fraud-detection-vae-augmentation
