"""
Génère Figure 2 : Curriculum Reward Function
"""
import matplotlib.pyplot as plt
import numpy as np

# Définir la reward function
def curriculum_reward(sr, target=0.5, sigma=0.2):
    """
    sr: success rate (0 to 1)
    target: target success rate
    sigma: standard deviation
    """
    if sr < 0.1:
        return 0.0
    else:
        return np.exp(-((sr - target)**2) / (2 * sigma**2))

# Générer les données
success_rates = np.linspace(0, 1, 200)
rewards = [curriculum_reward(sr) for sr in success_rates]

# Créer le graphique
fig, ax = plt.subplots(figsize=(8, 5))

# Courbe principale
ax.plot(success_rates, rewards, linewidth=3, color='#2c3e50', label=r'$f(\text{SR})$')

# Remplir sous la courbe
ax.fill_between(success_rates, 0, rewards, alpha=0.3, color='#3498db')

# Marquer les zones
ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, 
          label='Target SR = 0.5', alpha=0.7)
ax.axvline(x=0.1, color='red', linestyle=':', linewidth=2,
          label='Minimum SR = 0.1', alpha=0.7)

# Annotations
ax.annotate('Too Easy\n(Low Reward)', xy=(0.85, 0.2), fontsize=11,
           ha='center',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

ax.annotate('Optimal Zone\n(High Reward)', xy=(0.5, 1.05), fontsize=11,
           ha='center',
           bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))

ax.annotate('Too Hard\n(No Reward)', xy=(0.05, 0.2), fontsize=11,
           ha='center',
           bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

# Flèches explicatives
ax.annotate('', xy=(0.5, 1.0), xytext=(0.5, 0.85),
           arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Labels et titre
ax.set_xlabel('Success Rate (SR)', fontsize=13, fontweight='bold')
ax.set_ylabel('Curriculum Reward', fontsize=13, fontweight='bold')
ax.set_title('Curriculum Reward Function Design', fontsize=15, fontweight='bold', pad=15)

# Grille
ax.grid(True, alpha=0.3, linestyle='--')

# Légende
ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Limites
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.15)

# Equation
equation_text = r'$R_{\text{curriculum}}(\theta) = \begin{cases} \exp\left(-\frac{(\text{SR} - 0.5)^2}{2 \cdot 0.2^2}\right) & \text{if SR} \geq 0.1 \\ 0 & \text{otherwise} \end{cases}$'
ax.text(0.98, 0.65, equation_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('figure2_reward_function.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure2_reward_function.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2 sauvegardée: figure2_reward_function.pdf")
plt.show()
