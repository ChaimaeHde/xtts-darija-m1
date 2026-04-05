"""
Interface Gradio pour tester le modèle XTTS-v2 Darija M1
Le modèle est chargé depuis HuggingFace automatiquement
"""

import os, gc, torch, numpy as np
import soundfile as sf
import gradio as gr
from huggingface_hub import hf_hub_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

HF_REPO_ID = "ChaimaeHde/xtts-darija-m1"
MODEL_CACHE = "/tmp/xtts_darija_m1/"

# ── Chargement du modèle (une seule fois) ───────────────
model_inf   = None
config_inf  = None

def load_model_once():
    global model_inf, config_inf
    if model_inf is not None:
        return True
    try:
        os.makedirs(MODEL_CACHE, exist_ok=True)
        print("⬇️ Téléchargement depuis HuggingFace...")
        model_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="model.pth",  local_dir=MODEL_CACHE)
        config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json", local_dir=MODEL_CACHE)
        vocab_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab.json",  local_dir=MODEL_CACHE)

        config_inf = XttsConfig()
        config_inf.load_json(config_path)

        model_inf = Xtts.init_from_config(config_inf)
        model_inf.load_checkpoint(
            config_inf,
            checkpoint_path=model_path,
            vocab_path=vocab_path,
            eval=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_inf.to(device)
        print(f"✅ Modèle chargé sur {device}")
        return True
    except Exception as e:
        print(f"❌ Erreur chargement modèle: {e}")
        return False


# ── Fonction de génération ───────────────────────────────
def generate_darija_tts(text, ref_audio):
    if not text or not text.strip():
        return None, "❌ Texte vide"
    if ref_audio is None:
        return None, "❌ Audio de référence manquant"

    if not load_model_once():
        return None, "❌ Impossible de charger le modèle"

    try:
        sr, audio_data = ref_audio
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0:
                audio_data = audio_data / 32768.0

        ref_path = "/tmp/ref_input.wav"
        sf.write(ref_path, audio_data, sr)

        outputs = model_inf.synthesize(
            text=text, config=config_inf,
            speaker_wav=ref_path, language="ar",
        )

        out_path = "/tmp/output_darija.wav"
        sf.write(out_path, outputs["wav"], 24000)
        return out_path, "✅ Audio généré avec succès"

    except Exception as e:
        return None, f"❌ Erreur: {str(e)}"


# ── Interface Gradio ─────────────────────────────────────
with gr.Blocks(title="TTS Darija Marocain — M1") as demo:
    gr.Markdown("# 🎙️ TTS Darija Marocain")
    gr.Markdown(
        "Synthèse vocale pour le dialecte marocain (Darija) — "
        "XTTS-v2 fine-tuné sur DODa (Locuteur M1, 35 min)\n\n"
        "**Note :** Le modèle se charge depuis HuggingFace au premier lancement (~2 min)."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Texte en Darija (arabe)",
                placeholder="مرحبا، كيف داير؟",
                lines=3,
            )
            ref_audio_input = gr.Audio(
                label="Audio de référence (voix à cloner — 6-10 secondes)",
                type="numpy",
                sources=["upload", "microphone"],
            )
            btn = gr.Button("🎙️ Générer", variant="primary")

        with gr.Column():
            audio_output  = gr.Audio(label="Voix générée", type="filepath")
            status_output = gr.Textbox(label="Statut")

    btn.click(
        fn=generate_darija_tts,
        inputs=[text_input, ref_audio_input],
        outputs=[audio_output, status_output],
    )

    gr.Examples(
        examples=[
            ["مرحبا، كيف داير؟ واش كلشي مزيان؟"],
            ["الجو مزيان بزاف اليوم، خرجنا نتفرجو"],
            ["واش نتا مزيان؟ شنو كاين الجديد؟"],
            ["ما فهمتش، قول ليا مرة أخرى"],
            ["الله يحفظك، بارك الله فيك"],
        ],
        inputs=[text_input],
    )

    gr.Markdown(
        "---\n"
        "**Modèle :** [ChaimaeHde/xtts-darija-m1](https://huggingface.co/ChaimaeHde/xtts-darija-m1) | "
        "**Dataset :** [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset) | "
        "**Base :** XTTS-v2 (Coqui TTS)"
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=False)
