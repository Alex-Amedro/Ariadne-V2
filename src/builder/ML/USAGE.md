# 🚀 Guide de démarrage rapide

## Première utilisation

### Méthode 1 : Interface graphique (RECOMMANDÉ)

```powershell
cd src/builder/ML
python launcher.py
```

Un menu interactif s'affiche. Choisissez l'option souhaitée !

### Méthode 2 : Ligne de commande

```powershell
cd src/builder/ML

# Nouvel entraînement rapide
python train_coevolution.py --epochs 5 --timesteps 10000

# Visualiser les résultats
python visualize_training.py --list
python visualize_training.py --run runs/run_XXXXXX

# Voir l'agent jouer
python train_coevolution.py --load runs/run_XXXXXX --visualize-only
```

## Workflow typique

### 1️⃣ Entraîner

```powershell
# Avec le launcher
python launcher.py
> Choix: 1 (Nouvel entraînement)
> Choix: 2 (Normal - 20 époques)

# OU en ligne de commande
python train_coevolution.py --epochs 20 --visualize
```

### 2️⃣ Analyser

```powershell
# Lister les runs
python visualize_training.py --list

# Voir les graphiques
python visualize_training.py --run runs/run_XXXXXX
```

### 3️⃣ Regarder l'agent

```powershell
# Slow motion sur 5 niveaux
python train_coevolution.py --load runs/run_XXXXXX --visualize-only
```

### 4️⃣ Continuer si besoin

```powershell
# Charger et ajouter 10 époques
python train_coevolution.py --load runs/run_XXXXXX --epochs 10
```

## 📊 Interpréter les résultats

### Success Rate (Taux de succès)
- **40-60%** = Parfait ! 🎯
- **<20%** = Trop dur, l'agent apprend mal ❌
- **>80%** = Trop facile, pas challengeant ❌

### Diversity (Diversité)
- **>0.8** = Excellent ! Beaucoup de variété 🌟
- **0.5-0.8** = Correct ✅
- **<0.5** = Problème, trop similaire ⚠️

### Graphiques à surveiller
- **Success Rate** doit converger vers 50%
- **Diversity** doit rester élevée
- **Mean Reward** doit augmenter

## ⚙️ Configurations recommandées

### Test rapide (5 min)
```powershell
python train_coevolution.py --epochs 5 --timesteps 10000 --initial-timesteps 20000
```

### Entraînement normal (30 min)
```powershell
python train_coevolution.py --epochs 20 --timesteps 50000 --initial-timesteps 100000
```

### Entraînement intensif (2h)
```powershell
python train_coevolution.py --epochs 50 --timesteps 100000 --initial-timesteps 200000
```

## 🔧 Problèmes courants

### "ModuleNotFoundError: No module named 'stable_baselines3'"
→ Activer l'environnement virtuel :
```powershell
cd C:\Users\Fretz\Desktop\Taiwan\Ariadne-V2
.\rl_env\Scripts\activate
cd src\builder\ML
```

### L'agent ne s'améliore pas
→ Augmenter les timesteps :
```powershell
python train_coevolution.py --epochs 20 --timesteps 100000 --initial-timesteps 200000
```

### Tous les niveaux sont identiques (diversity=0)
→ C'est corrigé maintenant ! Vérifier que `generator.py` utilise bien `_init_weights()`

### Erreur de mémoire
→ Réduire le batch size dans le code (passer de 16 à 8)

## 💡 Astuces

1. **Toujours commencer par un test rapide** pour vérifier que tout fonctionne
2. **Sauvegarder régulièrement** : les modèles sont auto-sauvegardés toutes les 5 époques
3. **Comparer plusieurs runs** pour trouver les meilleurs hyperparamètres
4. **Utiliser le launcher** pour une expérience plus simple

## 📞 Aide

Pour plus de détails, voir `README.md`

Bon entraînement ! 🎮🤖
