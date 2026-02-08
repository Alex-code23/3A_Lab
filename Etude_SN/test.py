import matplotlib.pyplot as plt
import math

# --- Constantes du Papier [cite: 125, 128, 132, 134] ---
BATTERY_CAPACITY = 700.0  # Wh (ou Joules selon interprétation temporelle, ici unité abstraite W par itération)
K_FIXED = 10.0            # Coût opérationnel fixe (Ko)
K_CENTRALIZED = 5.0       # Coût comm. Centralisé (par drone)
K_HIERARCHICAL = 3.0      # Coût comm. Hiérarchique (facteur racine carrée)
K_HOLONIC = 1.0           # Coût comm. Holonique (constant)

# Seuils d'activation [cite: 136]
MIN_HIERARCHICAL = 14
MIN_HOLONIC = 42

class DroneSwarm:
    def __init__(self, architecture_mode):
        self.drones_battery = [] # Liste des niveaux de batterie
        self.architecture_mode = architecture_mode
        self.history_size = []
        self.history_energy = []
        self.history_connectivity = [] # 1.0 = connecté, 0.0 = déconnecté
        self.iteration = 0

    def calculate_energy_cost(self, n_drones):
        """
        Calcule le coût énergétique par drone selon l'architecture active.
        Basé sur les formules mathématiques du papier section 3.
        """
        # Détermination de l'architecture active
        active_arch = self.architecture_mode
        
        if self.architecture_mode == "adaptive":
            # Logique "LLM": Choisir l'architecture valide avec le coût le plus bas
            costs = {}
            
            # Centralisé toujours possible
            costs['centralized'] = K_FIXED + K_CENTRALIZED * n_drones
            
            # Hiérarchique si N >= 14
            if n_drones >= MIN_HIERARCHICAL:
                costs['hierarchical'] = K_FIXED + K_HIERARCHICAL * math.sqrt(n_drones)
            else:
                costs['hierarchical'] = float('inf')
                
            # Holonique si N >= 42
            if n_drones >= MIN_HOLONIC:
                costs['holonic'] = K_FIXED + K_HOLONIC
            else:
                costs['holonic'] = float('inf')
            
            # Sélection de la clé avec la valeur minimale
            active_arch = min(costs, key=costs.get)

        # Calcul du coût selon l'architecture choisie
        cost = 0.0
        connected = True

        if active_arch == "centralized":
            cost = K_FIXED + K_CENTRALIZED * n_drones
        
        elif active_arch == "hierarchical":
            if n_drones >= MIN_HIERARCHICAL:
                cost = K_FIXED + K_HIERARCHICAL * math.sqrt(n_drones)
            else:
                # "Drones cannot communicate" -> Coût de base uniquement
                cost = K_FIXED 
                connected = False

        elif active_arch == "holonic":
            if n_drones >= MIN_HOLONIC:
                cost = K_FIXED + K_HOLONIC
            else:
                # "Drones cannot communicate"
                cost = K_FIXED
                connected = False
        
        return cost, connected

    def step(self):
        # 1. Ajout de 2 nouveaux drones (batterie pleine) [cite: 138]
        self.drones_battery.extend([BATTERY_CAPACITY, BATTERY_CAPACITY])
        n_current = len(self.drones_battery)

        # 2. Calcul du coût pour cette itération
        cost_per_drone, is_connected = self.calculate_energy_cost(n_current)
        
        # 3. Mise à jour des batteries
        # Si non connecté, le papier implique que la mission échoue ou le drone consomme juste le fixe.
        # Ici on applique la consommation calculée.
        self.drones_battery = [b - cost_per_drone for b in self.drones_battery]

        # 4. Suppression des drones épuisés
        alive_drones = [b for b in self.drones_battery if b > 0]
        n_alive = len(alive_drones)
        self.drones_battery = alive_drones

        # Enregistrement des stats
        self.history_size.append(n_alive)
        self.history_energy.append(cost_per_drone * n_current if is_connected else 0) # Total energy swarm
        self.history_connectivity.append(1 if is_connected else 0)
        self.iteration += 1

def run_simulation():
    modes = ["centralized", "hierarchical", "holonic", "adaptive"]
    results = {}
    iterations = 125 # Pour matcher l'axe X de la Figure 3

    plt.figure(figsize=(14, 6))

    # --- Plot 1: Scalability (Taille de l'essaim vs Iteration) ---
    plt.subplot(1, 2, 1)
    
    for mode in modes:
        sim = DroneSwarm(mode)
        for _ in range(iterations):
            sim.step()
        results[mode] = sim
        plt.plot(sim.history_size, label=mode, linewidth=2)

    plt.title("Reproduction Fig 3(a): Scalability")
    plt.xlabel("Iteration")
    plt.ylabel("Swarm Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Energy Consumption per Iteration ---
    plt.subplot(1, 2, 2)
    # Note: Le papier plotte parfois la consommation totale ou moyenne. 
    # La Fig 4(a) montre une consommation qui monte puis se stabilise.
    # Pour 'Adaptive', le papier dit ~1036W median.
    
    for mode in modes:
        # On lisse un peu pour la lisibilité comme sur les graphs du papier
        plt.plot(results[mode].history_energy, label=mode, linewidth=1.5)

    plt.title("Reproduction Fig 4(a): Energy Consumption")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (W)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # --- Vérification Mathématique (Table 2 & 3) ---
    print(f"{'Mode':<15} | {'Max Size':<10} | {'Final Size':<10} | {'Connected It.'}")
    print("-" * 55)
    for mode, sim in results.items():
        max_size = max(sim.history_size)
        final_size = sim.history_size[-1]
        
        # Trouver quand la connectivité commence (première itération connectée)
        try:
            first_connect = sim.history_connectivity.index(1)
        except ValueError:
            first_connect = -1
            
        print(f"{mode:<15} | {max_size:<10} | {final_size:<10} | {first_connect}")

if __name__ == "__main__":
    run_simulation()