# 🗺️ ROADMAP ARIADNE-V2 - Co-Evolution System

**Date de début :** 29 Décembre 2025  
**Status actuel :** Phase 3 - Comparaisons et Extensions  
**Objectif final :** Papier complet 15-20 pages pour GT

---

## ✅ PHASE 1 : FONDATIONS (TERMINÉ)

### ✅ Environnement Paramétrique Custom
- [x] `parametric_minigrid.py` : Environnement MiniGrid paramétrable
- [x] Paramètres : grid_size, num_obstacles, num_doors, num_keys
- [x] Reward shaping : goal_reached, steps_penalty, exploration_bonus
- [x] Validation complète

### ✅ Générateur Neural
- [x] `generator.py` : MLP qui génère des paramètres de niveaux
- [x] Architecture : 8→64→64→4 + Sigmoid/Clipping
- [x] Input : vecteur latent aléatoire (8D)
- [x] Output : grid_size, num_obstacles, num_doors, num_keys

### ✅ Agent PPO
- [x] Stable-Baselines3 PPO
- [x] MlpPolicy
- [x] Optimisé pour MiniGrid

---

## ✅ PHASE 2 : CO-EVOLUTION MVP (TERMINÉ)

### ✅ Boucle de Co-évolution
- [x] `train_coevolution.py` : Système complet
- [x] Phase 1 : Génération batch de niveaux
- [x] Phase 2 : Entraînement agent (50k timesteps/epoch)
- [x] Phase 3 : Entraînement générateur (gradient sur perf agent)
- [x] Phase 4 : Logging complet
- [x] **RUN VALIDÉ :** 20 epochs, 80% success rate final

### ✅ Bug Fixes
- [x] IndexError dans co-évolution (validation batch_size)
- [x] Visualization freeze (plt.show() bloquant)
- [x] Tous les tests passent ✅

---

## ✅ PHASE 3 : MÉTRIQUES & VISUALISATIONS (TERMINÉ)

### ✅ Analyse Complète
- [x] `analyze_results.py` : 570 lignes, 4 graphiques PNG
  - Success rate évolution (+60% improvement)
  - Learning phases identification
  - Convergence analysis (16/20 epochs ≥40%)
  - Generator analysis (diversity 69.4%)
  
- [x] `create_dashboard.py` : Dashboard 8 panels
  - KPIs visuels
  - Distributions, timelines, scatter plots
  
- [x] `generate_html_report.py` : Rapport interactif
  - Navigation, CSS, images embedded
  - Tables de stats, configuration
  
- [x] `README_ANALYSE.md` : Interprétation complète
  - Résultats, insights, recommandations

### ✅ Transfer Learning
- [x] `test_transfer.py` : Test sur 7 environnements MiniGrid standards
- [x] **RÉSULTAT :** 100% sur Empty-5x5, 0% ailleurs
- [x] **INSIGHT :** Compétences spécifiques à l'environnement paramétrique
- [x] Graphiques et rapport générés

---

## 🔄 PHASE 3.5 : COMPARAISONS (EN COURS)

### 🔄 Baseline Comparison
- [x] `train_baseline.py` : Entraînement sur niveaux ALÉATOIRES
  - ✅ Script créé (250 lignes)
  - 🔄 **EN COURS :** Entraînement 20 epochs (~2-3h)
  - ⏳ Génération de runs/baseline_XXXXXX/
  
- [x] `compare_results.py` : Comparaison statistique
  - ✅ Script créé (400 lignes)
  - ⏳ Graphiques de comparaison (2 figures PNG)
  - ⏳ Test statistique (t-test, Cohen's d)
  - ⏳ Rapport texte + JSON

### ⏳ À FAIRE APRÈS BASELINE
```bash
# Quand baseline training terminé :
python compare_results.py \
  --coevol runs/run_20251229_035928 \
  --baseline runs/baseline_XXXXXX \
  --output comparison_results
```

**Attendu :**
- Co-évolution > Baseline (hypothèse)
- Différence statistiquement significative
- Figures pour papier

---

## 📅 PHASE 4 : EXTENSIONS & AMÉLIORATIONS

### 🎯 Extension A : Ablation Studies (PRIORITÉ 1)
**Objectif :** Montrer que chaque composant est important

**À implémenter :**
1. **Sans reward shaping** (reward binaire seulement)
   - Modifier parametric_minigrid.py
   - Entraîner 10-20 epochs
   - Comparer avec baseline

2. **Sans gradient générateur** (niveaux aléatoires constants)
   - Générateur fixe (pas d'update)
   - Seulement agent s'améliore
   - Comparer avec co-évolution

3. **Différentes architectures générateur**
   - Plus profond (8→128→128→64→4)
   - Plus simple (8→32→4)
   - Comparer performances

**Script à créer :** `run_ablations.py`

**Temps estimé :** 3-4 jours  
**Pages ajoutées :** +2-3 pages

---

### 🎯 Extension B : Diversity Objective (PRIORITÉ 2)
**Objectif :** Éviter niveaux tous identiques

**Méthode 1 : Novelty Search**
```python
# Ajouter au loss du générateur
novelty_loss = -diversity_metric(generated_levels)
total_loss = agent_performance_loss + lambda_novelty * novelty_loss
```

**Méthode 2 : MAP-Elites (Quality-Diversity)**
```python
# Grid 2D : (grid_size, num_obstacles)
archive = {}  # {(cell_x, cell_y): best_level}

for level in generated_levels:
    cell = compute_cell(level)
    if level.fitness > archive[cell].fitness:
        archive[cell] = level  # Garder le meilleur par cellule
```

**Métriques diversity :**
- Variance des paramètres (déjà implémenté)
- Distance moyenne entre niveaux
- Nombre de cellules couvertes (MAP-Elites)

**Script à créer :** `train_diversity.py`

**Temps estimé :** 5-7 jours  
**Pages ajoutées :** +3-4 pages

---

### 🎯 Extension C : Curriculum Pacing (PRIORITÉ 3)
**Objectif :** Contrôler la vitesse de progression

**Stratégies :**
1. **Adaptive pacing**
   ```python
   if agent_success_rate < 0.3:
       # Trop dur, ralentir
       difficulty_weight *= 0.9
   elif agent_success_rate > 0.7:
       # Trop facile, accélérer
       difficulty_weight *= 1.1
   ```

2. **Staged curriculum**
   ```python
   if epoch < 10:
       target_difficulty = 'easy'  # grid_size 5-7
   elif epoch < 30:
       target_difficulty = 'medium'  # grid_size 7-10
   else:
       target_difficulty = 'hard'  # grid_size 10-12
   ```

**Script à créer :** `train_curriculum.py`

**Temps estimé :** 3-4 jours  
**Pages ajoutées :** +2 pages

---

### 🎯 Extension D : Multi-Agent Training (AVANCÉ)
**Objectif :** Plusieurs agents en parallèle

**Concept :**
- Population d'agents (n=5-10)
- Générateur crée des niveaux difficiles pour TOUS
- Sélection naturelle des meilleurs agents

**Avantage :**
- Généralisation accrue
- Robustesse

**Temps estimé :** 7-10 jours  
**Pages ajoutées :** +3-4 pages

---

### 🎯 Extension E : PAIRED Comparison (SOTA)
**Objectif :** Comparer avec état de l'art

**Méthode :**
- Implémenter PAIRED (Protagonist-Antagonist Induced Regret)
- Comparer avec notre système
- Benchmarks standardisés

**Référence :**
- Dennis et al. (2020) - "Emergent Complexity via Multi-Agent Competition"

**Temps estimé :** 10-14 jours  
**Pages ajoutées :** +4-5 pages

---

## 📊 ÉTAT DES DONNÉES

### ✅ Runs Disponibles
```
runs/run_20251229_035928/  (Co-Evolution, 20 epochs)
├── logs/history.json       ✅ Success rate 20%→80%
├── models/
│   ├── agent_final.zip     ✅ Agent entraîné
│   ├── agent_best.zip      ✅ Meilleur (epoch 16, 86.7%)
│   └── generator_*.pth     ✅ Générateurs sauvegardés
└── analysis/               ✅ 8 fichiers d'analyse

transfer_results/           ✅ Transfer learning
├── transfer_results.json   ✅ Résultats sur 7 envs
├── transfer_performance.png ✅ Graphiques
└── transfer_report.txt     ✅ Rapport

runs/baseline_XXXXXX/       🔄 EN COURS
└── (en attente...)
```

---

## 📝 STRUCTURE PAPIER (15-20 pages)

### 1. Introduction (2 pages) ⏳
**À écrire :**
- Motivation : Pourquoi PCG + RL ?
- Problème : Curriculum design manuel coûteux
- Contribution : Co-évolution automatique
- Résultats clés : +60% amélioration, 80% success rate

### 2. Related Work (2-3 pages) ⏳
**Sections :**
- Procedural Content Generation (PCG)
- Curriculum Learning in RL
- Adversarial Environment Design (PAIRED, PLR)
- Co-evolution in RL

**Références clés :**
- PAIRED (Dennis et al. 2020)
- PLR (Jiang et al. 2021)
- Teacher-Student (Portelas et al. 2020)

### 3. Methodology (4-5 pages) ✅ (données prêtes)
**3.1 Custom MiniGrid Environment**
- Paramètres : grid_size, obstacles, doors, keys
- Reward function design
- Validation

**3.2 Generator Architecture**
- MLP 8→64→64→4
- Input : latent vector
- Output : level parameters

**3.3 Co-Evolution Training Loop**
```
for epoch in range(epochs):
    1. Generate batch (generator)
    2. Train agent (PPO, 50k timesteps)
    3. Train generator (gradient on agent perf)
    4. Log metrics
```

**3.4 Evaluation Protocol**
- Success rate
- Diversity metrics
- Transfer learning tests

### 4. Experiments (5-7 pages) 🔄 (en cours)
**4.1 Setup**
- Hyperparameters
- Hardware (GPU, CPU)
- Training time

**4.2 Main Results** ✅
- Success rate evolution (+60%)
- Convergence analysis
- Generator diversity (69.4%)

**4.3 Baseline Comparison** 🔄
- Co-evolution vs Random levels
- Statistical tests (t-test)
- Effect size (Cohen's d)

**4.4 Transfer Learning** ✅
- Test sur 7 environnements MiniGrid
- Résultats : spécialisation vs généralisation

**4.5 Ablation Studies** ⏳
- Sans reward shaping
- Sans gradient générateur
- Différentes architectures

### 5. Analysis (2-3 pages) ⏳
**5.1 Visualizations**
- Niveaux générés (grilles visuelles)
- Espace latent (t-SNE, PCA)
- Trajectoires agent

**5.2 Emergent Behaviors**
- Phases d'apprentissage
- Stratégies découvertes
- Générateur adversarial

**5.3 Discussion**
- Limitations (transfer limité)
- Insights théoriques
- Implications pratiques

### 6. Conclusion (1 page) ⏳
- Summary des contributions
- Limitations actuelles
- Future work (diversity, PAIRED, multi-agent)

---

## 🛠️ OUTILS & STACK TECHNIQUE

### ✅ Déjà Utilisé
- **Python 3.10** + Virtual env
- **Stable-Baselines3** : PPO agent
- **PyTorch** : Générateur neural
- **MiniGrid** : Environnements base
- **Matplotlib** : Visualisations
- **Scipy** : Statistiques
- **Seaborn** : Graphiques avancés
- **NumPy** : Calculs
- **JSON** : Logs et configs

### ⏳ À Ajouter
- **TensorBoard** : Tracking en temps réel
- **Weights & Biases** (optionnel) : Logging cloud
- **imageio** : Vidéos de l'agent
- **scikit-learn** : t-SNE, PCA pour visualiser espace latent
- **pandas** : Manipulation données pour analyses

---

## ⚡ ACTIONS IMMÉDIATES

### 🔴 CETTE SEMAINE
1. ✅ **Transfer learning** - TERMINÉ
2. 🔄 **Baseline training** - EN COURS (2-3h restantes)
3. ⏳ **Comparison analysis** - Dès que baseline terminé
4. ⏳ **Créer ablation studies script**
5. ⏳ **Lancer ablations** (sans reward shaping en priorité)

### 🟡 SEMAINE PROCHAINE
1. Diversity objective (Novelty Search)
2. Curriculum pacing
3. Vidéos de l'agent (imageio)
4. Visualisation espace latent (t-SNE)

### 🟢 SEMAINES 3-4
1. Écriture papier (Introduction, Related Work)
2. MAP-Elites implementation
3. PAIRED comparison (si temps)
4. Multi-agent training (si temps)

---

## 📈 MÉTRIQUES CLÉS ACTUELLES

### Co-Evolution (20 epochs)
```
Success Rate : 20% → 80% (+60%)
Best Score   : 86.7%
Mean SR      : 59.3% ± 22.1%
Diversity    : 69.4% ± 12%
Convergence  : 16/20 epochs ≥40%
```

### Transfer Learning
```
Empty-5x5    : 100% ✅
Empty-8x8    : 0%
DoorKey-5x5  : 0%
MultiRoom-N2 : 0%
Overall      : 14.3% (spécialisation)
```

### Baseline (en attente)
```
🔄 EN COURS...
Attendu : SR < Co-evolution
```

---

## 🎯 OBJECTIFS FINAUX

### Minimum Viable Paper (12-15 pages)
- ✅ Méthodologie claire
- ✅ Résultats co-évolution validés
- 🔄 Baseline comparison
- ⏳ 1-2 ablations
- ⏳ Visualisations complètes

### Full Paper (15-20 pages)
- ✅ MVP +
- ⏳ Diversity objective
- ⏳ Curriculum pacing
- ⏳ 3-4 ablations
- ⏳ PAIRED comparison (optionnel)

### Publication Targets
- **ICLR 2026** (deadline Mai 2026) - FAISABLE ✅
- **NeurIPS 2026** (deadline Mai 2026) - FAISABLE ✅
- **ICML 2026** (deadline Février 2026) - SERRÉ ⚠️

---

## 📞 CONTACT & NEXT STEPS

**Current Status :** Phase 3.5 - Baseline training en cours

**Next Meeting :**
- Analyser résultats baseline vs co-évolution
- Décider priorités extensions (diversity vs ablations)
- Planning écriture papier

**Questions Ouvertes :**
1. Diversity objective = priorité ou après ablations ?
2. PAIRED comparison = nécessaire ou nice-to-have ?
3. Target conference = ICLR ou NeurIPS ?

---

**Dernière mise à jour :** 1 Janvier 2026  
**Version :** 1.0
