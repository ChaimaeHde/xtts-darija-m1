"""
Génération audio avec le modèle XTTS-v2 fine-tuné sur Darija M1
Le modèle est téléchargé automatiquement depuis HuggingFace
"""

import os, gc, torch, json
import soundfile as sf
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

HF_REPO_ID = "ChaimaeHde/xtts-darija-m1"


def load_model(cache_dir="/content/xtts_m1_model"):
    """Télécharge et charge le modèle depuis HuggingFace."""
    os.makedirs(cache_dir, exist_ok=True)

    print("⬇️ Téléchargement du modèle depuis HuggingFace...")

    # Télécharger les fichiers du modèle fine-tuné
    model_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pth",  local_dir=cache_dir)
    config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json", local_dir=cache_dir)
    vocab_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab.json",  local_dir=cache_dir)

    print("✅ Fichiers téléchargés")

    # Charger le modèle
    config_inf = XttsConfig()
    config_inf.load_json(config_path)

    model = Xtts.init_from_config(config_inf)
    model.load_checkpoint(config_inf, checkpoint_path=model_path, vocab_path=vocab_path, eval=True)
    model.cuda() if torch.cuda.is_available() else model.cpu()

    print("✅ Modèle chargé")
    return model, config_inf


def generate_speech(model, config, text, speaker_wav, output_path="output.wav", language="ar"):
    """Génère un fichier audio depuis du texte en Darija."""
    outputs = model.synthesize(
        text=text, config=config,
        speaker_wav=speaker_wav, language=language,
    )
    sf.write(output_path, outputs["wav"], 24000)
    print(f"✅ Audio généré : {output_path}")
    return output_path


if __name__ == "__main__":
    model, config = load_model()
    generate_speech(
        model, config,
        text="مرحبا، كيف داير؟ واش كلشي مزيان معك اليوم؟",
        speaker_wav="ref_audio.wav",
        output_path="output_darija.wav"
    )
