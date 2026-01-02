# 🚀 GUIDE D'UTILISATION - ARIADNE-V2

Tous les scripts prêts à l'emploi pour vos expériences.

---

## 📁 STRUCTURE DES FICHIERS

```
src/builder/ML/
├── Core Files
│   ├── parametric_minigrid.py     # Environnement paramétrable
│   ├── generator.py               # Générateur neural (MLP)
│   ├── train_coevolution.py       # Co-évolution complète ✅
│   
├── Baselines & Ablations
│   ├── train_baseline.py          # Baseline (niveaux aléatoires) ✅
│   ├── run_ablations.py           # Ablation studies ✅
│   
├── Extensions
│   ├── train_diversity.py         # Diversity objective (Novelty Search) ✅
│   ├── train_curriculum.py        # Curriculum pacing adaptatif ✅
│   
├── Analysis & Visualization
│   ├── analyze_results.py         # Analyse complète d'un run ✅
│   ├── create_dashboard.py        # Dashboard visuel ✅
│   ├── generate_html_report.py    # Rapport HTML interactif ✅
│   ├── compare_results.py         # Comparaison 2 runs ✅
│   ├── compare_all.py             # Comparaison N runs ✅
│   ├── visualize_latent.py        # t-SNE, PCA, espace latent ✅
│   ├── record_agent.py            # Vidéos MP4 de l'agent ✅
│   ├── test_transfer.py           # Transfer learning ✅
│   
└── Documentation
    ├── ROADMAP.md                 # Plan complet du projet ✅
    └── COMMANDS.md                # Ce fichier ✅
```

---

## ⚡ COMMANDES RAPIDES

### 1️⃣ ENTRAÎNEMENT CO-ÉVOLUTION (Baseline)
```bash
cd C:\Users\Fretz\Desktop\Taiwan\Ariadne-V2
.\rl_env\Scripts\activate
cd src\builder\ML

# Entraînement standard (20 epochs)
python train_coevolution.py --epochs 20 --timesteps 50000

# Entraînement long (50 epochs)
python train_coevolution.py --epochs 50 --timesteps 50000

# Avec plus de samples par epoch
python train_coevolution.py --epochs 20 --timesteps 100000
```

**Output:** `runs/run_YYYYMMDD_HHMMSS/`

---

### 2️⃣ BASELINE (Niveaux Aléatoires)
```bash
# Baseline standard
python train_baseline.py --epochs 20 --timesteps 50000 --initial-timesteps 100000

# Baseline rapide (10 epochs)
python train_baseline.py --epochs 10 --timesteps 30000 --initial-timesteps 50000
```

**Output:** `runs/baseline_YYYYMMDD_HHMMSS/`

---

### 3️⃣ ABLATION STUDIES

#### A. Sans Reward Shaping
```bash
python run_ablations.py --ablation no_reward_shaping --epochs 15 --timesteps 50000
```

#### B. Sans Gradient Générateur (Fixe)
```bash
python run_ablations.py --ablation no_generator_gradient --epochs 15 --timesteps 50000
```

#### C. Générateur Simple (8→32→4)
```bash
python run_ablations.py --ablation simple_generator --epochs 15 --timesteps 50000
```

#### D. Générateur Complexe (8→128→128→64→4)
```bash
python run_ablations.py --ablation complex_generator --epochs 15 --timesteps 50000
```

#### E. Toutes les Ablations
```bash
python run_ablations.py --ablation all --epochs 15 --timesteps 50000
```

**Output:** `runs/ablation_TYPE_YYYYMMDD_HHMMSS/`

---

### 4️⃣ DIVERSITY OBJECTIVE (Novelty Search)
```bash
# Diversity standard
python train_diversity.py --epochs 20 --timesteps 50000 --diversity-weight 0.3

# Diversity forte
python train_diversity.py --epochs 20 --timesteps 50000 --diversity-weight 0.5 --archive-size 150

# Diversity faible
python train_diversity.py --epochs 20 --timesteps 50000 --diversity-weight 0.1
```

**Output:** `runs/diversity_YYYYMMDD_HHMMSS/`

---

### 5️⃣ CURRICULUM PACING

#### A. Adaptive Pacing (recommandé)
```bash
# Target SR = 50%
python train_curriculum.py --epochs 20 --timesteps 50000 --strategy adaptive --target-sr 0.5

# Target SR = 60% (plus agressif)
python train_curriculum.py --epochs 20 --timesteps 50000 --strategy adaptive --target-sr 0.6
```

#### B. Staged Curriculum (paliers fixes)
```bash
python train_curriculum.py --epochs 25 --timesteps 50000 --strategy staged
```

**Output:** `runs/curriculum_STRATEGY_YYYYMMDD_HHMMSS/`

---

## 📊 ANALYSE & VISUALISATION

### 6️⃣ ANALYSE COMPLÈTE D'UN RUN
```bash
# Analyser un run spécifique
python analyze_results.py

# Le script trouve automatiquement le dernier run
# Génère: 4 PNG + REPORT.txt + statistics.json
```

**Output:** `runs/run_XXXXXX/analysis/`

---

### 7️⃣ DASHBOARD VISUEL
```bash
python create_dashboard.py

# Génère un dashboard 8-panels en une image
```

**Output:** `runs/run_XXXXXX/analysis/00_DASHBOARD.png`

---

### 8️⃣ RAPPORT HTML INTERACTIF
```bash
python generate_html_report.py

# Ouvre automatiquement dans le navigateur
```

**Output:** `runs/run_XXXXXX/analysis/RAPPORT_COMPLET.html`

---

### 9️⃣ COMPARAISON 2 RUNS
```bash
# Comparer co-évolution vs baseline
python compare_results.py \
  --coevol runs/run_20251229_035928 \
  --baseline runs/baseline_20260101_151704 \
  --output comparison_results

# Génère: 2 PNG + statistics.json + rapport.txt
```

**Output:** `comparison_results/`

---

### 🔟 COMPARAISON GLOBALE (N runs)
```bash
# Comparer tous les runs ensemble
python compare_all.py \
  --runs \
    "Co-Evolution:runs/run_20251229_035928" \
    "Baseline:runs/baseline_20260101_151704" \
    "No Reward:runs/ablation_no_reward_shaping_XXXXX" \
    "Diversity:runs/diversity_XXXXX" \
  --output global_comparison

# Génère: 3 PNG + rapport complet + stats JSON
```

**Output:** `global_comparison/`

---

### 1️⃣1️⃣ TRANSFER LEARNING
```bash
# Tester sur environnements MiniGrid standards
python test_transfer.py \
  --run runs/run_20251229_035928 \
  --episodes 20 \
  --output transfer_results

# Teste sur: Empty-5x5, Empty-8x8, DoorKey-5x5/6x6/8x8, MultiRoom-N2/N4
```

**Output:** `transfer_results/`

---

### 1️⃣2️⃣ VISUALISATION ESPACE LATENT (t-SNE, PCA)
```bash
# Visualiser l'espace latent du générateur
python visualize_latent.py \
  --generator runs/run_20251229_035928/models/generator_final.pth \
  --output latent_viz \
  --samples 1000

# Génère: t-SNE, PCA, parameter space (3 PNG)
```

**Output:** `latent_viz/`

---

### 1️⃣3️⃣ ENREGISTREMENT VIDÉOS
```bash
# Mode standard (10 vidéos)
python record_agent.py \
  --agent runs/run_20251229_035928/models/agent_final.zip \
  --generator runs/run_20251229_035928/models/generator_final.pth \
  --output videos \
  --num-episodes 10 \
  --fps 10

# Mode showcase (easy/medium/hard)
python record_agent.py \
  --agent runs/run_20251229_035928/models/agent_final.zip \
  --generator runs/run_20251229_035928/models/generator_final.pth \
  --output videos_showcase \
  --showcase \
  --num-episodes 9 \
  --fps 10

# Sans générateur (niveaux random)
python record_agent.py \
  --agent runs/baseline_XXXXX/models/agent_final.zip \
  --output videos_baseline \
  --num-episodes 10
```

**Output:** `videos/*.mp4` + métadonnées JSON

---

## 🔄 WORKFLOWS COMPLETS

### 🎯 WORKFLOW 1: Expérience Complète (Co-évolution + Baseline)
```bash
# 1. Entraîner co-évolution
python train_coevolution.py --epochs 20 --timesteps 50000

# 2. Entraîner baseline
python train_baseline.py --epochs 20 --timesteps 50000 --initial-timesteps 100000

# 3. Comparer les deux
python compare_results.py \
  --coevol runs/run_XXXXXX \
  --baseline runs/baseline_XXXXXX \
  --output comparison

# 4. Analyser chacun
cd runs/run_XXXXXX
python ../../analyze_results.py
python ../../generate_html_report.py

cd ../baseline_XXXXXX
python ../../analyze_results.py
```

---

### 🎯 WORKFLOW 2: Ablation Study Complete
```bash
# 1. Entraîner toutes les ablations
python run_ablations.py --ablation all --epochs 15 --timesteps 50000

# 2. Comparer tous ensemble
python compare_all.py \
  --runs \
    "Co-Evolution:runs/run_20251229_035928" \
    "No Reward:runs/ablation_no_reward_shaping_XXXXX" \
    "No Gradient:runs/ablation_no_generator_gradient_XXXXX" \
    "Simple Gen:runs/ablation_simple_generator_XXXXX" \
    "Complex Gen:runs/ablation_complex_generator_XXXXX" \
  --output ablation_comparison
```

---

### 🎯 WORKFLOW 3: Paper-Ready Figures
```bash
# 1. Co-évolution (déjà fait)
# runs/run_20251229_035928

# 2. Analyses visuelles
python analyze_results.py
python create_dashboard.py
python generate_html_report.py

# 3. Transfer learning
python test_transfer.py --run runs/run_20251229_035928 --episodes 20

# 4. Espace latent
python visualize_latent.py \
  --generator runs/run_20251229_035928/models/generator_final.pth \
  --samples 1000

# 5. Vidéos showcase
python record_agent.py \
  --agent runs/run_20251229_035928/models/agent_final.zip \
  --generator runs/run_20251229_035928/models/generator_final.pth \
  --showcase \
  --num-episodes 9

# 6. Comparaison avec baseline
python compare_results.py \
  --coevol runs/run_20251229_035928 \
  --baseline runs/baseline_XXXXX \
  --output comparison_paper

# RÉSULTAT: Toutes les figures pour le papier ! 📝
```

---

## 📦 DÉPENDANCES

Si un script échoue avec "module not found":

```bash
.\rl_env\Scripts\activate

# Dépendances principales (déjà installées)
pip install torch stable-baselines3 gymnasium minigrid

# Pour analyses
pip install scipy seaborn matplotlib numpy

# Pour vidéos
pip install imageio imageio-ffmpeg

# Pour visualisations avancées
pip install scikit-learn pandas
```

---

## 🐛 TROUBLESHOOTING

### Erreur: "Il faut au moins autant de clés que de portes"
**Fix:** Déjà corrigé dans train_baseline.py (ligne 70)

### Erreur: "enable_reward_shaping not found"
**Fix:** Déjà ajouté dans parametric_minigrid.py (ligne 34)

### Baseline training lent
**Astuce:** Réduire `--epochs` ou `--timesteps`

### Out of memory
**Astuce:** Réduire `--batch-size` dans les trainers

### Vidéos ne s'enregistrent pas
**Fix:** `pip install imageio imageio-ffmpeg`

---

## 📈 MÉTRIQUES ACTUELLES

### ✅ Co-Evolution (20 epochs)
```
Success Rate : 20% → 80% (+60%)
Best Score   : 86.7%
Mean SR      : 59.3% ± 22.1%
Diversity    : 69.4% ± 12%
```

### ✅ Transfer Learning
```
Empty-5x5    : 100%
Empty-8x8    : 0%
Overall      : 14.3%
```

### 🔄 Baseline
```
EN COURS... (~2-3h)
```

---

## 🎯 PROCHAINES ÉTAPES RECOMMANDÉES

1. **Attendre baseline** (~2h) puis comparer
2. **Lancer ablation** `no_reward_shaping`
3. **Visualiser espace latent** (t-SNE, PCA)
4. **Enregistrer vidéos** pour présentation
5. **Diversity training** (20 epochs)

---

## 📞 AIDE RAPIDE

**Trouver le dernier run:**
```powershell
Get-ChildItem runs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

**Lister tous les runs:**
```powershell
Get-ChildItem runs | Select-Object Name, LastWriteTime | Format-Table
```

**Vérifier progression baseline:**
```powershell
Get-Content runs\baseline_20260101_151704\logs\history.json
```

---

**Dernière mise à jour:** 1 Janvier 2026  
**Version:** 1.0  
**Status:** Tous les scripts créés et prêts ! ✅
