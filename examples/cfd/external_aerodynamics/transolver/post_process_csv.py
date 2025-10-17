import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from pathlib import Path

# Read the train.yaml configuration
with open('conf/train.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get train and val data paths from config
train_path = config['data']['train']['data_path']
val_path = config['data']['val']['data_path']

# Convert relative paths to absolute paths relative to this script's location
script_dir = Path(__file__).parent
train_dir = (script_dir / train_path).resolve()
val_dir = (script_dir / val_path).resolve()

# Get list of zarr directories and extract the ID portions
# Handle both old format: "run_airFoil2D_SST_71.134_7.877_1.724_2.818_11.267.zarr"
# And new format: "airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32.zarr"
train_ids = {d.name.replace('run_', '').replace('.zarr', '') for d in train_dir.iterdir() if d.is_dir()}
val_ids = {d.name.replace('run_', '').replace('.zarr', '') for d in val_dir.iterdir() if d.is_dir()}

print(f"Found {len(train_ids)} train samples")
print(f"Found {len(val_ids)} val samples")

# Read the CSV file
df = pd.read_csv("residuals_and_errors_non_dim_2d.csv")

# Compute total momentum residuals (combining x and y components)
# Since momentum is a vector, we sum the absolute values of x and y components
df['total_momentum_pred'] = df['total_momentum_x_pred'] + df['total_momentum_y_pred']
df['total_momentum_true'] = df['total_momentum_x_true'] + df['total_momentum_y_true']

# Classify each row as train or val based on filename
# CSV filenames can be:
# Old format: "pred_internal_airFoil2D_SST_71.134_7.877_1.724_2.818_11.267.vtu"
# New format: "pred_airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32_internal.vtu"
def classify_split(filename):
    # Extract the ID portion from filename
    # Handle both "pred_internal_" prefix and "pred_..._internal.vtu" format
    file_id = filename.replace('pred_', '').replace('_internal', '').replace('.vtu', '')
    if file_id in train_ids:
        return 'train'
    elif file_id in val_ids:
        return 'val'
    else:
        return 'unknown'

df['split'] = df['filename'].apply(classify_split)

print(f"\nData distribution:")
print(df['split'].value_counts())

# Save the dataframe with split column to a new CSV file
df.to_csv('residuals_and_errors_non_dim_2d_with_split.csv', index=False)
print(f"\nSaved CSV with split column")

# Split data into train and val
df_train = df[df['split'] == 'train'].copy()
df_val = df[df['split'] == 'val'].copy()

# Sort each dataframe and assign IDs
df_train = df_train.sort_values('total_continuity_pred', ascending=True).reset_index(drop=True)
df_train['ID'] = range(len(df_train))

df_val = df_val.sort_values('total_continuity_pred', ascending=True).reset_index(drop=True)
df_val['ID'] = range(len(df_val))

# Get numeric columns
numeric_columns = [col for col in df.columns if col not in ['filename', 'ID', 'split', 'total_momentum_z_pred', 'total_momentum_z_true']]

# Create line plots with train and val side by side
n_cols = len(numeric_columns)
fig, axes = plt.subplots(n_cols, 2, figsize=(20, 3*n_cols))

for idx, col in enumerate(numeric_columns):
    # Train plot
    ax_train = axes[idx, 0]
    ax_train.plot(df_train['ID'], df_train[col], marker='o', markersize=3, linewidth=1, color='#1f77b4')
    ax_train.set_ylabel(col, fontsize=10)
    ax_train.grid(True, alpha=0.3)
    ax_train.set_xlabel('ID (sorted by total_continuity_pred)')
    if idx == 0:
        ax_train.set_title('Train', fontsize=12, fontweight='bold')
    
    # Val plot
    ax_val = axes[idx, 1]
    ax_val.plot(df_val['ID'], df_val[col], marker='o', markersize=3, linewidth=1, color='#ff7f0e')
    ax_val.set_ylabel(col, fontsize=10)
    ax_val.grid(True, alpha=0.3)
    ax_val.set_xlabel('ID (sorted by total_continuity_pred)')
    if idx == 0:
        ax_val.set_title('Val', fontsize=12, fontweight='bold')

fig.suptitle('Residuals and Errors: Train vs Val', fontsize=14, fontweight='bold', y=1.0)
fig.tight_layout()
fig.savefig('residuals_and_errors_plot_non_dim_2d.png', dpi=150, bbox_inches='tight')

# Create correlation plots with train and val (Pearson and Spearman)
fig_corr, axes_corr = plt.subplots(2, 2, figsize=(28, 24))

# Train correlations
corr_pearson_train = df_train[numeric_columns].corr(method='pearson')
corr_spearman_train = df_train[numeric_columns].corr(method='spearman')

sns.heatmap(corr_pearson_train, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=axes_corr[0, 0])
axes_corr[0, 0].set_title('Train - Pearson Correlation', fontsize=14, pad=20, fontweight='bold')

sns.heatmap(corr_spearman_train, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=axes_corr[1, 0])
axes_corr[1, 0].set_title('Train - Spearman Correlation', fontsize=14, pad=20, fontweight='bold')

# Val correlations
corr_pearson_val = df_val[numeric_columns].corr(method='pearson')
corr_spearman_val = df_val[numeric_columns].corr(method='spearman')

sns.heatmap(corr_pearson_val, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=axes_corr[0, 1])
axes_corr[0, 1].set_title('Val - Pearson Correlation', fontsize=14, pad=20, fontweight='bold')

sns.heatmap(corr_spearman_val, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=axes_corr[1, 1])
axes_corr[1, 1].set_title('Val - Spearman Correlation', fontsize=14, pad=20, fontweight='bold')

plt.tight_layout()
fig_corr.savefig('correlation_plot_non_dim_2d.png', dpi=150, bbox_inches='tight')

print(f"Train samples: {len(df_train)}")
print(f"Val samples: {len(df_val)}")

