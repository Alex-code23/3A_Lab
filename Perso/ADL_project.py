import numpy as np
import matplotlib.pyplot as plt
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

sigma = 0.25  
n_augments = 5
X_train_aug = np.vstack([X_train + np.random.normal(0, sigma, X_train.shape) for _ in range(n_augments)])
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

# ==========================================
# 3. Test : Accuracy vs n (Moyenne et Variance)
# ==========================================
def generate_perturbations(X, radius):
    directions = np.random.randn(*X.shape)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    return X + (directions / norms) * radius

radii = [0.0, 0.2, 0.5, 0.8] 
n_values = [50, 100, 500, 1000, 2500, 5000, 10000]

# Nombre de répétitions pour estimer l'espérance et la variance
num_trials = 5 

plt.figure(figsize=(10, 6))

for radius in radii:
    # Stockage des accuracies pour calculer les statistiques
    accuracies_trials = {n: [] for n in n_values}
    
    for trial in range(num_trials):
        # Nouvelle perturbation aléatoire à chaque essai pour une estimation non biaisée
        X_test_perturbed = generate_perturbations(X_test, radius)
        
        for n in n_values:
            preds = predict_g(X_test_perturbed, f_model, n_samples=n, sigma=sigma)
            acc = accuracy_score(y_test, preds)
            accuracies_trials[n].append(acc)
            
    # Calcul de la moyenne et de l'écart-type empiriques
    mean_acc = np.array([np.mean(accuracies_trials[n]) for n in n_values])
    std_acc = np.array([np.std(accuracies_trials[n]) for n in n_values])
    
    # Tracé de la moyenne
    line = plt.plot(n_values, mean_acc, marker='o', label=f'Rayon $\delta$ = {radius}')
    color = line[0].get_color()
    
    # Tracé de la zone de variance (Moyenne ± Écart-type)
    plt.fill_between(n_values, 
                     np.clip(mean_acc - std_acc, 0, 1), # Borner entre 0 et 1 (intervalle valide pour accuracy)
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