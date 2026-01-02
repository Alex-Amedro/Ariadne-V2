"""
VISUALISATION AVANCÉE : t-SNE, PCA, espace latent du générateur.
Analyse de l'espace des niveaux générés.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import json

from generator import LevelGenerator


class LatentSpaceVisualizer:
    """Visualise l'espace latent du générateur."""
    
    def __init__(self, generator_path, save_dir='visualizations'):
        """
        Args:
            generator_path: Chemin vers le générateur (.pth)
            save_dir: Dossier de sauvegarde
        """
        self.generator = LevelGenerator(latent_dim=8)
        self.generator.load_state_dict(torch.load(generator_path))
        self.generator.eval()
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[LatentViz] Générateur chargé: {generator_path}")
        print(f"[LatentViz] Sauvegarde dans: {save_dir}")
    
    def sample_latent_space(self, num_samples=1000):
        """Échantillonne l'espace latent."""
        with torch.no_grad():
            z_samples = torch.randn(num_samples, 8)
            
            # Générer les paramètres pour chaque z
            levels = []
            for z in z_samples:
                params = self.generator.generate_level_params(z)
                levels.append(params)
        
        return z_samples.numpy(), levels
    
    def visualize_tsne(self, num_samples=1000):
        """Visualisation t-SNE de l'espace latent."""
        print("\n[t-SNE] Génération de l'embedding...")
        
        z, params, levels = self.sample_latent_space(num_samples)
        
        # t-SNE sur les paramètres normalisés
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        embedding = tsne.fit_transform(params)
        
        # Créer la figure avec 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('t-SNE Visualization of Generated Levels', fontsize=18, fontweight='bold')
        
        # 1. Couleur = grid_size
        ax = axes[0, 0]
        grid_sizes = [l['grid_size'] for l in levels]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=grid_sizes, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Grid Size')
        ax.set_title('Grid Size Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        # 2. Couleur = num_obstacles
        ax = axes[0, 1]
        obstacles = [l['num_obstacles'] for l in levels]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=obstacles, cmap='plasma', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Num Obstacles')
        ax.set_title('Obstacles Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        # 3. Couleur = num_doors
        ax = axes[1, 0]
        doors = [l['num_doors'] for l in levels]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=doors, cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Num Doors')
        ax.set_title('Doors Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
        
        # 4. Densité (heatmap)
        ax = axes[1, 1]
        heatmap, xedges, yedges = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(im, ax=ax, label='Density')
        ax.set_title('Density Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        save_path = self.save_dir / "tsne_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
    
    def visualize_pca(self, num_samples=1000):
        """Visualisation PCA de l'espace latent."""
        print("\n[PCA] Analyse en composantes principales...")
        
        z, params, levels = self.sample_latent_space(num_samples)
        
        # PCA
        pca = PCA(n_components=4)
        embedding = pca.fit_transform(params)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('PCA Analysis of Generated Levels', fontsize=18, fontweight='bold')
        
        # 1. PC1 vs PC2
        ax = axes[0, 0]
        grid_sizes = [l['grid_size'] for l in levels]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                           c=grid_sizes, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Grid Size')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('First Two Principal Components', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. PC3 vs PC4
        ax = axes[0, 1]
        obstacles = [l['num_obstacles'] for l in levels]
        scatter = ax.scatter(embedding[:, 2], embedding[:, 3],
                           c=obstacles, cmap='plasma', alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Obstacles')
        ax.set_xlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_ylabel(f'PC4 ({pca.explained_variance_ratio_[3]:.1%})')
        ax.set_title('Third and Fourth Principal Components', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Explained Variance
        ax = axes[1, 0]
        ax.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel('Principal Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 5))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Annoter
        for i, var in enumerate(pca.explained_variance_ratio_):
            ax.text(i+1, var + 0.01, f'{var:.1%}', ha='center', fontweight='bold')
        
        # 4. Cumulative Variance
        ax = axes[1, 1]
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(range(1, 5), cumsum, 'o-', linewidth=3, markersize=10, color='darkgreen')
        ax.fill_between(range(1, 5), 0, cumsum, alpha=0.3, color='green')
        ax.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
        ax.set_xlabel('Number of Components', fontsize=12)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 5))
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Annoter
        for i, var in enumerate(cumsum):
            ax.text(i+1, var + 0.02, f'{var:.1%}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / "pca_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
        
        # Rapport
        print(f"\nExplained Variance:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.2%}")
        print(f"  Total (4 PCs): {cumsum[-1]:.2%}")
    
    def visualize_parameter_space(self, num_samples=1000):
        """Visualise l'espace des paramètres 2D (tous les pairs)."""
        print("\n[ParamSpace] Visualisation de l'espace des paramètres...")
        
        z, params, levels = self.sample_latent_space(num_samples)
        
        # Extraire les paramètres
        grid_sizes = np.array([l['grid_size'] for l in levels])
        obstacles = np.array([l['num_obstacles'] for l in levels])
        doors = np.array([l['num_doors'] for l in levels])
        keys = np.array([l['num_keys'] for l in levels])
        
        # Créer pairplot
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Parameter Space Coverage', fontsize=20, fontweight='bold')
        
        params_dict = {
            'Grid Size': grid_sizes,
            'Obstacles': obstacles,
            'Doors': doors
        }
        
        param_names = list(params_dict.keys())
        
        for i, param_y in enumerate(param_names):
            for j, param_x in enumerate(param_names):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(params_dict[param_y], bins=20, alpha=0.7, 
                           color='steelblue', edgecolor='black')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{param_y} Distribution', fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                else:
                    # Off-diagonal: scatter
                    ax.scatter(params_dict[param_x], params_dict[param_y],
                              alpha=0.4, s=10, c='navy')
                    ax.set_xlabel(param_x)
                    ax.set_ylabel(param_y)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / "parameter_space.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVED] {save_path}")
        plt.close()
    
    def analyze_all(self, num_samples=1000):
        """Lance toutes les visualisations."""
        print("="*60)
        print("LATENT SPACE ANALYSIS")
        print("="*60)
        
        self.visualize_tsne(num_samples)
        self.visualize_pca(num_samples)
        self.visualize_parameter_space(num_samples)
        
        print("\n" + "="*60)
        print("ANALYSIS TERMINÉE!")
        print("="*60)
        print(f"\nFichiers générés dans: {self.save_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualisation de l'espace latent")
    parser.add_argument("--generator", type=str, required=True,
                       help="Chemin vers le générateur (.pth)")
    parser.add_argument("--output", type=str, default="visualizations",
                       help="Dossier de sortie")
    parser.add_argument("--samples", type=int, default=1000,
                       help="Nombre de samples")
    
    args = parser.parse_args()
    
    viz = LatentSpaceVisualizer(args.generator, args.output)
    viz.analyze_all(args.samples)
