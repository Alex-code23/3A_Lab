import numpy as np
import matplotlib.pyplot as plt

def get_gradient(theta, beta, loss_type="quadratic"):
    if loss_type == "quadratic":
        return beta * theta
    elif loss_type == "logcosh":
        return beta * np.tanh(theta)
    elif loss_type == "nonconvex":
        return (beta / 2.0) * np.sin(2.0 * theta)
    else:
        raise ValueError("Unknown loss type")

def run_simulation(beta, eta, T, delta, sigma, sigma_tilde, loss_type="quadratic", num_runs=500):
    theta_clean = np.zeros((num_runs, T))
    theta_poison = np.zeros((num_runs, T))
    
    for t in range(T - 1):
        noise_clean = np.random.normal(0, sigma, num_runs)
        noise_poison = np.random.normal(0, sigma_tilde, num_runs)
        
        g_clean = get_gradient(theta_clean[:, t], beta, loss_type) + noise_clean
        g_poison = get_gradient(theta_poison[:, t], beta, loss_type) + delta + noise_poison
        
        theta_clean[:, t+1] = theta_clean[:, t] - eta * g_clean
        theta_poison[:, t+1] = theta_poison[:, t] - eta * g_poison

    empirical_divergence = np.mean(np.abs(theta_poison - theta_clean), axis=0)
    
    E_norm_noise = (sigma + sigma_tilde) * np.sqrt(2 / np.pi)
    K = np.abs(delta) + E_norm_noise
    
    time_steps = np.arange(T)
    geom_bound = (K / beta) * ((1 + eta * beta)**time_steps - 1)
    exp_bound = (K / beta) * (np.exp(eta * beta * time_steps) - 1)
    
    return time_steps, empirical_divergence, geom_bound, exp_bound

def plot_eta_delta_grid():
    beta, T, sigma, sigma_tilde = 0.1, 150, 0.1, 0.1
    
    # Configurations : croisement de eta (LR) et delta (Attack)
    configs = [
        {"eta": 0.01, "delta": 0.1, "title": "Low LR, Weak Attack"},
        {"eta": 0.01, "delta": 1.0, "title": "Low LR, Strong Attack"},
        {"eta": 0.1,  "delta": 0.1, "title": "High LR, Weak Attack"},
        {"eta": 0.1,  "delta": 1.0, "title": "High LR, Strong Attack"}
    ]
    
    losses = [
        ("quadratic", "Quadratic", "blue"),
        ("logcosh", "Log-Cosh", "green"),
        ("nonconvex", "Sine Squared", "purple")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, conf in enumerate(configs):
        eta = conf["eta"]
        delta = conf["delta"]
        ax = axes[idx]
        
        # Les bornes théoriques ne dépendent que des hyperparamètres, pas de la topologie
        t, _, geom_b, exp_b = run_simulation(beta, eta, T, delta, sigma, sigma_tilde, loss_type="quadratic")
        
        ax.plot(t, exp_b, label="Exponential Bound", linestyle=":", color='red', linewidth=2)
        ax.plot(t, geom_b, label="Exact Geometric Bound", linestyle="--", color='orange', linewidth=2)
        
        max_emp = 0
        for loss_type, label, color in losses:
            _, emp_div, _, _ = run_simulation(beta, eta, T, delta, sigma, sigma_tilde, loss_type=loss_type)
            ax.plot(t, emp_div, label=f"Emp: {label}", linewidth=2, color=color)
            max_emp = max(max_emp, np.max(emp_div))
            
        ax.set_title(f"{conf['title']} ($\\eta$={eta}, $\\delta$={delta})")
        ax.set_xlabel("Steps (t)")
        ax.set_ylabel("Accumulated Error $\\Delta_t$")
        ax.legend(loc="upper left")
        ax.grid(True)
        
        # Limite dynamique de l'axe Y pour préserver la lisibilité des divergences empiriques
        ax.set_ylim(0, max_emp * 2.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_eta_delta_grid()