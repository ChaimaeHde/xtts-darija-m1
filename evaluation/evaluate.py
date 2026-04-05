"""
Évaluation WER, CER et MOS du modèle XTTS-v2 Darija M1
"""

import os
import numpy as np
from faster_whisper import WhisperModel
from jiwer import wer, cer


def evaluate_wer_cer(audio_files, original_texts, whisper_model_size="large-v2"):
    """
    Calcule WER et CER pour une liste d'audios générés.

    Args:
        audio_files: liste de chemins vers les fichiers WAV générés
        original_texts: liste des textes originaux (même ordre)
        whisper_model_size: taille du modèle Whisper

    Returns:
        dict avec résultats par audio et moyennes
    """
    print(f"⏳ Chargement Whisper {whisper_model_size}...")
    asr_model = WhisperModel(whisper_model_size, device="cuda", compute_type="float16")
    print("✅ Whisper chargé")

    results = []
    for i, (audio_path, texte_original) in enumerate(zip(audio_files, original_texts)):
        segments, _ = asr_model.transcribe(audio_path, language="ar")
        texte_reconnu = " ".join([s.text for s in segments]).strip()

        score_wer = wer(texte_original, texte_reconnu)
        score_cer = cer(texte_original, texte_reconnu)

        results.append({
            "audio": os.path.basename(audio_path),
            "original": texte_original,
            "recognized": texte_reconnu,
            "WER": score_wer,
            "CER": score_cer,
        })

        print(f"\n--- Audio {i+1} ---")
        print(f"Original  : {texte_original}")
        print(f"Reconnu   : {texte_reconnu}")
        print(f"WER       : {score_wer:.2%}")
        print(f"CER       : {score_cer:.2%}")

    avg_wer = np.mean([r["WER"] for r in results])
    avg_cer = np.mean([r["CER"] for r in results])

    print("\n" + "="*50)
    print(f"📊 WER moyen : {avg_wer:.2%}")
    print(f"📊 CER moyen : {avg_cer:.2%}")

    return {"results": results, "avg_wer": avg_wer, "avg_cer": avg_cer}


def calculate_mos(scores_dict):
    """
    Calcule le MOS global depuis des scores humains.

    Args:
        scores_dict: dict {critère: [scores 1-5 par audio]}
    """
    print("\n📊 MOS RESULTS")
    print("=" * 40)
    mos_values = []
    for critere, valeurs in scores_dict.items():
        mos = np.mean(valeurs)
        mos_values.append(mos)
        print(f"{critere:20} : {mos:.2f} / 5")

    mos_global = np.mean(mos_values)
    print("=" * 40)
    print(f"{'MOS Global':20} : {mos_global:.2f} / 5")
    return mos_global


# Résultats réels du Test M1
RESULTS_M1 = {
    "avg_wer": 0.5077,
    "avg_cer": 0.2003,
    "details": [
        {"audio": 1, "original": "واش نتا مزيان؟ شنو كاين الجديد اليوم فالمغرب؟", "WER": 0.625, "CER": 0.1778},
        {"audio": 2, "original": "فين غادي نلقاو الحل ديال هاد المشكل؟",           "WER": 0.8571, "CER": 0.5278},
        {"audio": 3, "original": "الله يحفظك، بارك الله فيك",                      "WER": 0.0,    "CER": 0.0},
        {"audio": 4, "original": "ما فهمتش",                                        "WER": 0.5,    "CER": 0.375},
        {"audio": 5, "original": "فين كاين السوق؟",                                 "WER": 0.3333, "CER": 0.2},
        {"audio": 6, "original": "أنا عيان، بغيت ننعس شوية",                       "WER": 0.3333, "CER": 0.08},
        {"audio": 7, "original": "الجو مزيان بزاف اليوم، خرجنا نتفرجو",            "WER": 0.3333, "CER": 0.0571},
        {"audio": 8, "original": "خاصك ترد بالك لشنو كتقول قدام الناس",            "WER": 0.4286, "CER": 0.0857},
        {"audio": 9, "original": "ماكنتش غادي نخرج",                               "WER": 1.6667, "CER": 0.5},
        {"audio": 10,"original": "مرحبا كيف داير",                                  "WER": 0.0,    "CER": 0.0},
    ]
}
