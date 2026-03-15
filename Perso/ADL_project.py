import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Préparation des données et du modèle
# ==========================================
iris = load_iris()
X = iris.data[:, :2]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sigma = 0.2  
n_augments = 5
np.random.seed(42)
X_train_aug = np.vstack([X_train for _ in range(n_augments)])  # + np.random.normal(0, sigma, X_train.shape)
y_train_aug = np.hstack([y_train for _ in range(n_augments)])

f_model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
f_model.fit(X_train_aug, y_train_aug)

# ==========================================
# 2. Définition du classifieur lissé g
# ==========================================
def predict_g(X_input, model, n_samples, sigma):
    y_pred_g = []
    for x in X_input:
        epsilon = np.random.normal(0, sigma, (n_samples, X_input.shape[1]))
        x_noisy = x + epsilon
        preds_f = model.predict(x_noisy)
        counts = np.bincount(preds_f, minlength=3)
        y_pred_g.append(np.argmax(counts))
    return np.array(y_pred_g)

def predict_proba_g(X_input, model, n_samples, sigma, n_classes=3):
    """Calcule la distribution de probabilité empirique lissée en moyennant les probabilités de f."""
    P_g = np.zeros((X_input.shape[0], n_classes))
    
    for i, x in enumerate(X_input):
        # Génération du bruit Monte Carlo
        epsilon = np.random.normal(0, sigma, (n_samples, X_input.shape[1]))
        x_noisy = x + epsilon
        
        # Évaluation des probabilités continues (soft labels) par f
        probs_f = model.predict_proba(x_noisy)
        
        # Lissage par espérance mathématique empirique
        P_g[i, :] = np.mean(probs_f, axis=0)
        
    return P_g

# ==========================================
# 3. Test : Accuracy vs n (Moyenne et Variance)
# ==========================================
def generate_perturbations(X, radius):
    directions = np.random.randn(*X.shape)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    return X + (directions / norms) * radius

radii = [0.0, 0.2, 0.5, 0.8] 
n_values_acc = [1,3,5, 10, 50, 100, 500, 1000, 5000, 10000]

num_trials = 25 

plt.figure(figsize=(10, 6))

for radius in radii:
    print(f"Évaluation pour rayon de perturbation $\delta$ = {radius}...")
    accuracies_trials = {n: [] for n in n_values_acc}
    for trial in range(num_trials):
        X_test_perturbed = generate_perturbations(X_test, radius)
        for n in n_values_acc:
            preds = predict_g(X_test_perturbed, f_model, n_samples=n, sigma=sigma)
            acc = accuracy_score(y_test, preds)
            accuracies_trials[n].append(acc)
            
    mean_acc = np.array([np.mean(accuracies_trials[n]) for n in n_values_acc])
    std_acc = np.array([np.std(accuracies_trials[n]) for n in n_values_acc])
    
    line = plt.plot(n_values_acc, mean_acc, marker='o', label=f'Rayon $\delta$ = {radius}')
    color = line[0].get_color()
    plt.fill_between(n_values_acc, 
                     np.clip(mean_acc - std_acc, 0, 1), 
                     np.clip(mean_acc + std_acc, 0, 1), 
                     color=color, alpha=0.2)

plt.title("Précision de $g$ en fonction de $n$ et $\delta$ (Moyenne $\pm$ Écart-type)")
plt.xlabel("Nombre d'échantillons Monte Carlo ($n$)")
plt.ylabel("Accuracy")
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 3.5. Évolution de l'accuracy de f et g vs delta
# ==========================================
# Définition des paramètres pour la comparaison
n_samples_robustness = 100  # Nombre d'échantillons Monte Carlo fixé pour g
radii_robustness = np.linspace(0, 1.5, 15) # Vecteur des rayons de perturbation
num_trials_robustness = 10  # Répétitions pour réduire la variance empirique

print(f"Calcul des accuracies vs delta pour f et g (sigma={sigma}, n={n_samples_robustness})...")

mean_acc_f = []
mean_acc_g = []

for r in radii_robustness:
    trials_f = []
    trials_g = []
    for _ in range(num_trials_robustness):
        X_test_pert = generate_perturbations(X_test, r)
        
        # Évaluation empirique du modèle f
        preds_f = f_model.predict(X_test_pert)
        trials_f.append(accuracy_score(y_test, preds_f))
        
        # Évaluation empirique du modèle g
        preds_g = predict_g(X_test_pert, f_model, n_samples=n_samples_robustness, sigma=sigma)
        trials_g.append(accuracy_score(y_test, preds_g))
        
    mean_acc_f.append(np.mean(trials_f))
    mean_acc_g.append(np.mean(trials_g))

# Traçage de la figure
plt.figure(figsize=(9, 6))
plt.plot(radii_robustness, mean_acc_f, marker='s', linestyle='-', color='red', label='Modèle de base $f$')
plt.plot(radii_robustness, mean_acc_g, marker='o', linestyle='-', color='blue', label=f'Modèle lissé $g$ ($\sigma={sigma}, n={n_samples_robustness}$)')

plt.title("Dégradation de l'accuracy de $f$ et $g$ sous perturbation (Moyenne sur 10 essais)")
plt.xlabel("Rayon de perturbation $\delta$")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 4. Visualisation des frontières de décision
# ==========================================
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

grid_points = np.c_[xx.ravel(), yy.ravel()]

Z_f = f_model.predict(grid_points).reshape(xx.shape)
N_SAMPLES = 150
Z_g = predict_g(grid_points, f_model, n_samples=N_SAMPLES, sigma=sigma).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].contourf(xx, yy, Z_f, alpha=0.4, cmap=plt.cm.RdYlBu)
axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
axes[0].set_title("Classifieur de base $f$")
axes[0].set_xlabel("Longueur du sépale")
axes[0].set_ylabel("Largeur du sépale")

axes[1].contourf(xx, yy, Z_g, alpha=0.4, cmap=plt.cm.RdYlBu)
axes[1].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu)
axes[1].set_title(f"Classifieur lissé $g$ ($\sigma={sigma}, n={N_SAMPLES}$)")
axes[1].set_xlabel("Longueur du sépale")
axes[1].set_ylabel("Largeur du sépale")

plt.tight_layout()
plt.show()

# ==========================================
# 5. Calcul des métriques D et KL Divergence
# ==========================================
# Grille plus grossière pour limiter la complexité de calcul des heatmaps
xx_coarse, yy_coarse = np.meshgrid(np.arange(x_min, x_max, 0.15),
                                   np.arange(y_min, y_max, 0.15))
grid_coarse = np.c_[xx_coarse.ravel(), yy_coarse.ravel()]

# Base model predictions
Z_f_hard = f_model.predict(grid_coarse)
P_f_soft = f_model.predict_proba(grid_coarse)

# Hyperparamètres de la heatmap
sigma_values = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
n_values_hm = [10, 50, 100, 200, 500, 1000, 5000]

heatmap_D = np.zeros((len(sigma_values), len(n_values_hm)))
heatmap_KL = np.zeros((len(sigma_values), len(n_values_hm)))

eps = 1e-10 # Paramètre de régularisation pour éviter log(0)
P_f_soft_clipped = np.clip(P_f_soft, eps, 1.0)
P_f_soft_clipped /= P_f_soft_clipped.sum(axis=1, keepdims=True)

for i, sig in enumerate(sigma_values):
    for j, n_val in enumerate(n_values_hm):
        print(f"Calcul pour sigma={sig}, n={n_val}...")
        # 1. Taux de désaccord empirique (D)
        Z_g_hard = predict_g(grid_coarse, f_model, n_samples=n_val, sigma=sig)
        heatmap_D[i, j] = np.mean(Z_f_hard != Z_g_hard)
        
        # 2. Divergence KL
        P_g_soft = predict_proba_g(grid_coarse, f_model, n_samples=n_val, sigma=sig)
        P_g_soft_clipped = np.clip(P_g_soft, eps, 1.0)
        P_g_soft_clipped /= P_g_soft_clipped.sum(axis=1, keepdims=True)
        
        # Calcul KL pixel par pixel, puis moyenne spatiale
        kl_divs = np.sum(P_f_soft_clipped * np.log(P_f_soft_clipped / P_g_soft_clipped), axis=1)
        heatmap_KL[i, j] = np.mean(kl_divs)

# ==========================================
# 6. Affichage des Heatmaps
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap D(f,g)
sns.heatmap(heatmap_D, annot=True, fmt=".3f", cmap="YlOrRd", 
            xticklabels=n_values_hm, yticklabels=sigma_values, ax=axes[0],
            cbar_kws={'label': "Taux d'écart $D(f, g)$"})
axes[0].invert_yaxis()
axes[0].set_title("Désaccord spatial des frontières (Hard labels)")
axes[0].set_xlabel("Nombre d'échantillons Monte Carlo ($n$)")
axes[0].set_ylabel("Variance du lissage ($\sigma$)")

# Heatmap KL(f||g)
sns.heatmap(heatmap_KL, annot=True, fmt=".3f", cmap="magma", 
            xticklabels=n_values_hm, yticklabels=sigma_values, ax=axes[1],
            cbar_kws={'label': "Divergence $D_{KL}(P_f || P_g)$"})
axes[1].invert_yaxis()
axes[1].set_title("Divergence probabiliste $KL$ (Soft labels)")
axes[1].set_xlabel("Nombre d'échantillons Monte Carlo ($n$)")
axes[1].set_ylabel("Variance du lissage ($\sigma$)")

plt.tight_layout()
plt.show()