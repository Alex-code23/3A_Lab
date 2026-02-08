import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap

# --- 1. PARAMÈTRES ET CONSTANTES (Selon papier) ---
K_FIXED = 10.0
# Coûts comm: Central=5, Hierarch=3, Holonic=1
COST_ARCH = {0: 5.0, 1: 3.0, 2: 1.0} 

# Mappings (Enums)
ARCH_NAMES = {0: 'Centralized', 1: 'Hierarchical', 2: 'Holonic'}
TASK_NAMES = {0: 'Search & Rescue', 1: 'Large-Area Mapping', 2: 'Supply Delivery'}

# --- 2. ENTRAÎNEMENT DU MODÈLE CONTEXTUEL (TABLE 1) ---
def train_context_aware_model():
    # Features: [Task_Type, Swarm_Size_Cat, Comm_Qual, Failure_Risk]
    # Task: 0=SAR, 1=Map, 2=Deliv
    # Size: 0=Small, 1=Med, 2=Large
    # Comm: 0=Low, 1=Mod, 2=Good
    # Risk: 0=Low, 1=High
    
    # Données synthétiques basées STRICTEMENT sur Table 1
    X = [
        # SAR scenarios
        [0, 0, 2, 0], # SAR, Small, Good, Low -> Centralized (0)
        [0, 2, 0, 1], # SAR, Large, Low, High -> Holonic (2)
        [0, 1, 1, 0], # SAR, Med, Mod, Low -> Hierarchical (1) (Interpolé)
        
        # Mapping scenarios
        [1, 0, 2, 0], # Map, Small, Good, Low -> Hierarchical (1) (Note: Table 1 row 3 distinct)
        [1, 2, 1, 1], # Map, Large, Mod, Mod(High) -> Holonic (2)
        
        # Delivery scenarios
        [2, 1, 2, 0], # Deliv, Medium, Good, Low -> Hierarchical (1)
        [2, 2, 0, 1], # Deliv, Large, Low, High -> Holonic (2)
        
        # Etats critiques (Status-Based Recommendations)
        [0, 0, 0, 1], # Any, Small, Low, High (Critical Fail) -> Hierarchical (1)
        [1, 2, 0, 1], # Any, Large, Low, High (Overload) -> Holonic (2)
    ]
    y = [0, 2, 1, 1, 2, 1, 2, 1, 2] # Labels associés

    # On sur-échantillonne pour renforcer les règles
    X = np.array(X * 50)
    y = np.array(y * 50)

    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    clf.fit(X, y)
    return clf

# --- 3. DÉFINITION DES PROFILS DE MISSION (DYNAMIQUE) ---
class MissionProfile:
    def __init__(self, name, task_id):
        self.name = name
        self.task_id = task_id
        
    def get_environment(self, t, total_duration, n_drones):
        progress = t / total_duration
        
        # --- SCÉNARIO 1 : SEARCH AND RESCUE (SAR) ---
        # Début calme, puis catastrophe (perte signal, haut risque), puis stabilisation
        if self.task_id == 0:
            if progress < 0.3: # Phase approche
                comm = 2 # Good
                risk = 0 # Low
            elif progress < 0.7: # Phase zone crash (Interférences)
                comm = 0 # Low (Signal jam)
                risk = 1 # High (Debris)
            else: # Phase retour
                comm = 1 # Moderate
                risk = 0 # Low

        # --- SCÉNARIO 2 : MAPPING ---
        # Le risque est constant, mais la taille augmente massivement (besoin de scale)
        elif self.task_id == 1:
            comm = 2 if n_drones < 40 else 1 # Le réseau sature doucement
            risk = 0 # Mapping est généralement sûr
            
        # --- SCÉNARIO 3 : SUPPLY DELIVERY ---
        # Risque augmente linéairement avec la distance (temps)
        elif self.task_id == 2:
            comm = 1 # Moderate (Longue distance)
            risk = 1 if progress > 0.6 else 0 # Risque fatigue matériaux fin de vol

        # Mapping vers catégories discrètes pour le LLM
        # Size: <15=S, <50=M, >50=L
        size_cat = 0 if n_drones < 15 else (1 if n_drones < 50 else 2)
        
        return [self.task_id, size_cat, comm, risk]

# --- 4. MOTEUR DE SIMULATION ---
def run_simulation(mission_profile, model, duration=100):
    n_drones = 2 # Start small
    history_energy = []
    history_arch = []
    history_risk = []
    history_comm = []
    
    for t in range(duration):
        # 1. Obtenir l'état de l'environnement (Dynamique non-linéaire)
        features = mission_profile.get_environment(t, duration, n_drones)
        # features = [Task, Size_Cat, Comm, Risk]
        
        # 2. Décision du LLM
        arch_decision = model.predict([features])[0]
        
        # 3. Calcul Coût Energétique (Physique)
        # Coût de base
        cost = K_FIXED
        
        # Coût Communication
        if arch_decision == 0: # Centralized
            cost += COST_ARCH[0] * n_drones
        elif arch_decision == 1: # Hierarchical
            if n_drones >= 14: cost += COST_ARCH[1] * np.sqrt(n_drones)
            else: cost += 0 # Pas de comm possible
        elif arch_decision == 2: # Holonic
            if n_drones >= 42: cost += COST_ARCH[2] # Constant
            else: cost += 0 # Pas de comm
            
        # Pénalité de "Mauvaise Architecture"
        # Si Risque élevé (1) ET Architecture PAS Holonic (2) -> Surcoût de réparation/retransmission
        risk_val = features[3]
        if risk_val == 1 and arch_decision != 2:
            cost *= 1.5 # Pénalité: le système lutte contre l'environnement
            
        history_energy.append(cost)
        history_arch.append(arch_decision)
        history_risk.append(risk_val)
        history_comm.append(features[2])
        
        # Croissance essaim
        n_drones += 2
        
    return history_energy, history_arch, history_risk, history_comm

# --- 5. EXÉCUTION ET VISUALISATION ---
model = train_context_aware_model()

# Simulation de deux missions contrastées
mission_sar = MissionProfile("Search & Rescue (Crisis)", 0)
mission_map = MissionProfile("Mapping (Scaling)", 1)

res_sar = run_simulation(mission_sar, model)
res_map = run_simulation(mission_map, model)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4)

def plot_mission(ax_energy, ax_param, res, title):
    energy, archs, risks, comms = res
    iterations = range(len(energy))
    
    # Couleurs architecture
    cmap = ListedColormap(['orange', 'green', 'purple'])
    norm = plt.Normalize(0, 2)
    
    # Graphique Energie + Fond Architecture
    ax_energy.plot(iterations, energy, color='black', linewidth=1.5, label='Conso (W)')
    ax_energy.imshow([archs], aspect='auto', cmap=cmap, norm=norm, 
                     extent=[0, len(energy), 0, max(energy)*1.1], alpha=0.3)
    ax_energy.set_title(f"{title}: Architecture & Énergie")
    ax_energy.set_ylabel("Watts")
    ax_energy.set_xlabel("Temps / Taille Essaim")
    
    # Graphique Paramètres Environnementaux
    ax2 = ax_param.twinx()
    l1 = ax_param.plot(iterations, risks, 'r-', label='Risque (0/1)')
    l2 = ax2.plot(iterations, comms, 'b--', label='Qualité Comm (0-2)')
    
    ax_param.set_title(f"{title}: Contexte Environnemental")
    ax_param.set_ylabel("Niveau Risque", color='r')
    ax2.set_ylabel("Qualité Comm", color='b')
    ax_param.set_yticks([0, 1])
    ax2.set_yticks([0, 1, 2])
    
    # Légende unifiée
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax_param.legend(lns, labs, loc='center right')

plot_mission(axes[0,0], axes[0,1], res_sar, "Mission A: Search & Rescue")
plot_mission(axes[1,0], axes[1,1], res_map, "Mission B: Large Mapping")

# Légende globale pour les architectures
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='orange', alpha=0.5, label='Centralized'),
    Patch(facecolor='green', alpha=0.5, label='Hierarchical'),
    Patch(facecolor='purple', alpha=0.5, label='Holonic'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))

plt.show()

# =================================================================

# --- 1. RE-ENTRAÎNEMENT DU MODÈLE (Conforme Table 1) ---
# On reprend la logique stricte pour garantir la fidélité au papier
def get_trained_model():
    # Features: [Size_Cat, Comm_Qual, Failure_Risk]
    # Size: 0=Small(<15), 1=Med(<50), 2=Large(>50)
    # Comm: 0=Low, 1=Mod, 2=Good
    # Risk: 0=Low, 1=High
    
    X = [
        [0, 2, 0], [0, 2, 0], # Small, Good, Low -> Centralized (0)
        [1, 2, 0], [1, 1, 0], # Med, Good/Mod, Low -> Hierarchical (1)
        [2, 1, 0], [2, 0, 0], # Large, Mod/Low, Low -> Holonic (2)
        [0, 0, 1], [1, 0, 1], [2, 0, 1], # Any Size, Low Comm, High Risk -> Holonic (2) (Safety first)
        [2, 2, 0] # Large, Good, Low -> Hierarchical (1) (Optimization rule)
    ]
    y = [0, 0, 1, 1, 2, 2, 2, 2, 2, 1] 
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)
    return clf

model = get_trained_model()

# --- 2. GÉNÉRATION DE LA MATRICE DE DÉCISION ---
# On va scanner toutes les combinaisons possibles de Taille vs Qualité Comm
swarm_sizes = np.linspace(2, 150, 300) # De 2 à 150 drones
comm_quality = np.linspace(0, 2, 300)  # De 0 (Low) à 2 (Good)

# Grille de résultats
decision_map = np.zeros((len(comm_quality), len(swarm_sizes)))

# On fixe le Risque à "Faible" (0) pour voir la transition pure de scalabilité.
# Puis on fera une passe avec Risque "Élevé" (1).
RISK_SCENARIO = 0 

for i, comm in enumerate(comm_quality):
    for j, size in enumerate(swarm_sizes):
        # Discretisation pour le modèle (Mapping du papier)
        cat_size = 0 if size < 15 else (1 if size < 50 else 2)
        cat_comm = 0 if comm < 0.66 else (1 if comm < 1.33 else 2)
        
        # Prédiction
        decision = model.predict([[cat_size, cat_comm, RISK_SCENARIO]])[0]
        decision_map[i, j] = decision

# --- 3. VISUALISATION ---
plt.figure(figsize=(10, 8))
cmap = ListedColormap(['#FFA500', '#228B22', '#800080']) # Orange, Green, Purple

# Affichage de la Heatmap
# Origin 'lower' pour que 0 soit en bas
plt.imshow(decision_map, extent=[2, 150, 0, 2], origin='lower', aspect='auto', cmap=cmap, alpha=0.8)

# Annotations des zones
plt.text(10, 1.5, "CENTRALIZED\n(Zone de départ)", color='white', fontweight='bold')
plt.text(35, 1.0, "HIERARCHICAL\n(Zone de transition)", color='white', fontweight='bold', ha='center')
plt.text(100, 0.5, "HOLONIC\n(Zone de charge/crise)", color='white', fontweight='bold', ha='center')

# Labels et Titres
plt.title(f"Carte de Décision de l'Architecture (Risque = {'Faible' if RISK_SCENARIO==0 else 'Élevé'})")
plt.xlabel("Taille de l'Essaim (N)")
plt.ylabel("Qualité de Communication (0=Mauvaise, 2=Bonne)")

# Création d'une légende personnalisée
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FFA500', label='Centralized (Low Latency)'),
    Patch(facecolor='#228B22', label='Hierarchical (Balanced)'),
    Patch(facecolor='#800080', label='Holonic (High Resilience)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.grid(True, linestyle=':', color='white', alpha=0.5)
plt.show()