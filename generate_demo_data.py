"""Generate synthetic credit card data for demonstration."""
import pandas as pd
import numpy as np
import os

# Create synthetic credit card data
np.random.seed(42)
n_samples = 1000
n_frauds = 50

# Normal transactions
normal_amount = np.random.exponential(scale=50, size=n_samples - n_frauds)
normal_time = np.random.uniform(0, 86400, size=n_samples - n_frauds)

# Fraud transactions (higher amounts)
fraud_amount = np.random.exponential(scale=200, size=n_frauds)
fraud_time = np.random.uniform(0, 86400, size=n_frauds)

# Combine
amounts = np.concatenate([normal_amount, fraud_amount])
times = np.concatenate([normal_time, fraud_time])
labels = np.concatenate([np.zeros(n_samples - n_frauds), np.ones(n_frauds)])

# Add random features V1-V28
n_features = 28
features = np.random.randn(n_samples, n_features)
features[n_samples - n_frauds:, :] += np.random.randn(n_frauds, n_features) * 2

# Create DataFrame
df = pd.DataFrame(features, columns=[f'V{i+1}' for i in range(n_features)])
df['Time'] = times
df['Amount'] = amounts
df['Class'] = labels.astype(int)

# Save
os.makedirs('data', exist_ok=True)
df.to_csv('data/creditcard.csv', index=False)

print(f"✓ Created synthetic dataset: {n_samples} transactions ({n_frauds} frauds)")
print(f"✓ Shape: {df.shape}")
print(f"✓ File: data/creditcard.csv")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
