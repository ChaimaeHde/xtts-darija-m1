# 🎙️ XTTS-v2 Fine-tuning — Darija Marocaine (M1)

Fine-tuning du modèle XTTS-v2 de Coqui TTS sur la Darija Marocaine,  
en utilisant le locuteur masculin M1 du dataset DODa (35 minutes, 650 samples).

## 📊 Résultats

| Métrique | Test F1 (baseline) | Test M1 (ce projet) | Amélioration |
|----------|-------------------|---------------------|--------------|
| WER moyen | 82.04% | 50.77% | −31.3 pts ✅ |
| CER moyen | 38.49% | 20.03% | −18.5 pts ✅ |

## 🚀 Tester le modèle (Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChaimaeHde/xtts-darija-m1/blob/main/notebooks/demo_colab.ipynb)

ou 
Exécutez cette cellule sur colab:
```bash
!pip install -q coqui-tts gradio huggingface_hub soundfile
!git clone https://github.com/ChaimaeHde/xtts-darija-m1.git
%cd xtts-darija-m1
!python app.py

```
## 📦 Installation

```bash
pip install coqui-tts datasets soundfile gradio huggingface_hub
```

## 🎯 Utilisation rapide

```python
from interface.gradio_app import demo
demo.launch()
```

## 🏗️ Structure du projet

```
xtts-darija-m1/
├── app.py                    ← Interface Gradio principale (lancer ici)
├── config/
│   └── default_config.py     ← Chemins et hyperparamètres centralisés
├── data/
│   └── prepare_dataset.py    ← Pipeline de préparation des données
├── training/
│   └── finetune.py           ← Script de fine-tuning
├── inference/
│   └── generate.py           ← Génération audio
├── evaluation/
│   └── evaluate.py           ← WER, CER, MOS
├── interface/
│   └── gradio_app.py         ← Interface Gradio
└── notebooks/
    └── demo_colab.ipynb      ← Démo Colab complète
```

## 🔧 Dataset

- **Source** : [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)
- **Locuteur** : M1 (Male 1), indices 4000–4649
- **Durée** : ~35 minutes, 650 samples
- **Format** : LJSpeech (pipe-séparé)

## 🧠 Modèle

Le modèle fine-tuné est disponible sur HuggingFace :  
👉 [ChaimaeHde/xtts-darija-m1](https://huggingface.co/ChaimaeHde/xtts-darija-m1)

Fichiers disponibles :
- `model.pth` — Poids fine-tunés (~2GB)
- `config.json` — Configuration du modèle
- `vocab.json` — Vocabulaire du tokenizer

## 📈 Détails techniques

- **Base** : XTTS-v2 (Coqui TTS)
- **Epochs** : 5
- **Batch size** : 2 (avec mixed_precision=True)
- **Learning rate** : 5e-6
- **grad_accum_steps** : 126 → effective batch = 252
- **GPU** : Colab T4 (14.5 GB)

## 📝 Citation

```bibtex
@misc{xtts-darija-m1,
  title={XTTS-v2 Fine-tuning for Moroccan Darija — Male Speaker M1},
  author={ChaimaeHde},
  year={2026},
  url={https://github.com/ChaimaeHde/xtts-darija-m1}
}
```
