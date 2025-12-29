"""
Génère Figure 1 : Overview du système de co-évolution
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

# Couleurs
color_gen = '#3498db'  # Bleu
color_agent = '#e74c3c'  # Rouge
color_env = '#2ecc71'  # Vert

# 1. GENERATOR BOX
gen_box = FancyBboxPatch((0.5, 3.5), 2, 1.5, 
                         boxstyle="round,pad=0.1", 
                         edgecolor=color_gen, 
                         facecolor=color_gen, 
                         alpha=0.3,
                         linewidth=2)
ax.add_patch(gen_box)
ax.text(1.5, 4.5, 'Generator', ha='center', va='center', 
        fontsize=14, fontweight='bold', color=color_gen)
ax.text(1.5, 4.1, r'$g_\psi(z) \rightarrow \theta$', ha='center', va='center', 
        fontsize=10, style='italic')
ax.text(1.5, 3.8, 'Neural Network', ha='center', va='center', 
        fontsize=9, color='gray')

# Latent input
latent = Circle((0.3, 4.25), 0.15, facecolor='white', edgecolor=color_gen, linewidth=2)
ax.add_patch(latent)
ax.text(0.3, 4.25, r'$z$', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(0.3, 3.9, 'latent', ha='center', va='center', fontsize=7)

# Arrow from latent to generator
arrow_latent = FancyArrowPatch((0.45, 4.25), (0.5, 4.25),
                              arrowstyle='->', mutation_scale=15, 
                              linewidth=1.5, color=color_gen)
ax.add_patch(arrow_latent)

# 2. GENERATED LEVELS (middle)
level_y = 4.25
level_x_start = 3
for i, (size, color) in enumerate([(0.6, '#d4f4dd'), (0.7, '#b8e6c9'), (0.8, '#9cd9b5')]):
    x_pos = level_x_start + i * 1
    level = Rectangle((x_pos, level_y - size/2), size, size,
                     facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(level)
    
    # Draw mini grid inside
    n_cells = 4
    cell_size = size / n_cells
    for row in range(n_cells):
        for col in range(n_cells):
            if np.random.rand() > 0.7:  # Random obstacles
                mini_cell = Rectangle((x_pos + col*cell_size, level_y - size/2 + row*cell_size),
                                     cell_size, cell_size, facecolor='gray', alpha=0.5)
                ax.add_patch(mini_cell)
    
    # Goal marker
    goal = Circle((x_pos + size*0.8, level_y + size*0.3), 0.08, 
                 facecolor='yellow', edgecolor='orange', linewidth=1)
    ax.add_patch(goal)
    
    ax.text(x_pos + size/2, level_y - size/2 - 0.3, f'Env {i+1}', 
           ha='center', fontsize=8)

ax.text(4, 5.2, 'Generated Environments', ha='center', fontsize=11, 
       fontweight='bold', color=color_env)

# Arrow from generator to levels
arrow_gen_levels = FancyArrowPatch((2.5, 4.25), (3.0, 4.25),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2, color=color_gen)
ax.add_patch(arrow_gen_levels)
ax.text(2.75, 4.6, r'$\theta$', ha='center', fontsize=10, style='italic')

# 3. PPO AGENT BOX
agent_box = FancyBboxPatch((5.5, 3.5), 2, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor=color_agent,
                          facecolor=color_agent,
                          alpha=0.3,
                          linewidth=2)
ax.add_patch(agent_box)
ax.text(6.5, 4.5, 'PPO Agent', ha='center', va='center',
       fontsize=14, fontweight='bold', color=color_agent)
ax.text(6.5, 4.1, r'$\pi_\phi(s)$', ha='center', va='center',
       fontsize=10, style='italic')
ax.text(6.5, 3.8, 'Policy Network', ha='center', va='center',
       fontsize=9, color='gray')

# Arrow from levels to agent
arrow_levels_agent = FancyArrowPatch((5.0, 4.25), (5.5, 4.25),
                                    arrowstyle='->', mutation_scale=20,
                                    linewidth=2, color=color_agent)
ax.add_patch(arrow_levels_agent)
ax.text(5.25, 4.6, 'Train', ha='center', fontsize=9, style='italic')

# 4. FEEDBACK ARROW (bottom)
# Performance metrics box
perf_box = FancyBboxPatch((4, 1.5), 1.5, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='purple',
                         facecolor='lavender',
                         linewidth=1.5)
ax.add_patch(perf_box)
ax.text(4.75, 2.1, 'Performance', ha='center', fontsize=10, fontweight='bold')
ax.text(4.75, 1.8, 'Success Rate', ha='center', fontsize=8)

# Arrow from agent to performance
arrow_agent_perf = FancyArrowPatch((6.5, 3.5), (5.2, 2.3),
                                  arrowstyle='->', mutation_scale=15,
                                  linewidth=2, color='purple',
                                  linestyle='dashed')
ax.add_patch(arrow_agent_perf)
ax.text(6.2, 2.8, 'Evaluate', ha='center', fontsize=8, style='italic')

# Arrow from performance to generator (feedback)
arrow_perf_gen = FancyArrowPatch((4, 1.9), (1.5, 3.5),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2.5, color='orange')
ax.add_patch(arrow_perf_gen)
ax.text(2.3, 2.5, 'Curriculum\nReward', ha='center', fontsize=9,
       fontweight='bold', color='orange')

# 5. CO-EVOLUTION CYCLE INDICATOR
cycle_text = ax.text(4, 0.5, 'Co-Evolution Loop', ha='center',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Epoch indicator
ax.text(8.5, 5.5, 'Epoch t', ha='right', fontsize=11,
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.annotate('', xy=(8.5, 5.2), xytext=(8.5, 5.8),
           arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))

# Settings
ax.set_xlim(0, 9)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_aspect('equal')

plt.title('Adversarial Curriculum Generation System', 
         fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figure1_overview.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_overview.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1 sauvegardée: figure1_overview.pdf")
plt.show()
