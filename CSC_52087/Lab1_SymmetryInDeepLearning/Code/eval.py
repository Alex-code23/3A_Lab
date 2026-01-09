
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        x_batch = torch.tensor(X_test[i][j:j+batch_size], dtype=torch.long).to(device)
        y_pred_deepsets.append(deepsets(x_batch))
        y_pred_lstm.append(lstm(x_batch))
        
        
    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(y_test[i], np.round(y_pred_deepsets))
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(y_test[i], np.round(y_pred_lstm))
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


# Visualization to compare Deepsets and LSTM
plt.style.use('seaborn-v0_8-whitegrid') 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

styles = {
    'DeepSets': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'}, # Bleu
    'LSTM':     {'color': '#d62728', 'marker': 's', 'linestyle': '--'}  # Rouge
}

# --- Plot 1: Accuracy ---
ax1.plot(cards, results['deepsets']['acc'], label='DeepSets', **styles['DeepSets'], linewidth=2.5)
ax1.plot(cards, results['lstm']['acc'], label='LSTM', **styles['LSTM'], linewidth=2.5)

ax1.set_xlabel('Cardinality (Set Size)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy vs. Set Size', fontsize=14)
ax1.set_xticks(cards) # Force l'affichage de toutes les cardinalités
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(frameon=True, fontsize=11, loc='best')

# --- Plot 2: MAE ---
ax2.plot(cards, results['deepsets']['mae'], label='DeepSets', **styles['DeepSets'], linewidth=2.5)
ax2.plot(cards, results['lstm']['mae'], label='LSTM', **styles['LSTM'], linewidth=2.5)

ax2.set_xlabel('Cardinality (Set Size)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
ax2.set_title('Model MAE vs. Set Size', fontsize=14)
ax2.set_xticks(cards)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(frameon=True, fontsize=11, loc='best')

plt.suptitle("DeepSets vs LSTM: Performance Analysis on Sequence Summation", fontsize=16, y=1.02)
plt.tight_layout()

plt.savefig('results_comparison.png', dpi=300, bbox_inches='tight')
