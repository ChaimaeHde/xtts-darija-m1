"""
Script de fine-tuning XTTS-v2 pour la Darija Marocaine — Locuteur M1
"""

import gc, os, glob, shutil, threading, time, torch
from trainer import Trainer, TrainerArgs
from TTS.api import TTS
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples

# ─── Config ─────────────────────────────────────────
DATA_PATH       = "/content/doda_m1_35min/"
OUT_PATH        = "/content/pfa_outputs_male/"
MODEL_DIR       = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"
TRAINING_OUTPUT = OUT_PATH + "run/training/"
DRIVE_BEST      = "/content/drive/MyDrive/xtts_darija_m1_best/"

def setup_base_model():
    """Télécharge le modèle de base XTTS-v2 et les fichiers manquants."""
    import requests
    os.environ["COQUI_TOS_AGREED"] = "1"
    print("⬇️ Téléchargement modèle de base...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    del tts; gc.collect(); torch.cuda.empty_cache()

    HF = "https://huggingface.co/coqui/XTTS-v2/resolve/main/"
    for fname, url in [
        ("dvae.pth",      HF + "dvae.pth"),
        ("mel_stats.pth", HF + "mel_stats.pth"),
        ("mel_norms.pth", HF + "mel_stats.pth"),
        ("vocab.json",    HF + "vocab.json"),
    ]:
        dest = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(dest):
            print(f"⬇️ {fname}...")
            r = requests.get(url, stream=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024): f.write(chunk)
            print(f"✅ {fname}")


def load_dataset_config():
    """Charge la configuration du dataset et les samples."""
    dataset_config = BaseDatasetConfig(
        dataset_name="doda_m1", path=DATA_PATH,
        meta_file_train="train.csv", meta_file_val="",
        ignored_speakers=None, formatter="ljspeech", language="ar",
    )
    train_samples, eval_samples = load_tts_samples(
        [dataset_config], eval_split=True,
        eval_split_max_size=256, eval_split_size=0.1,
    )
    print(f"✅ Train: {len(train_samples)} | Eval: {len(eval_samples)}")
    return dataset_config, train_samples, eval_samples


def finetune(restore_path=None):
    """Lance le fine-tuning."""
    gc.collect()
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.makedirs(TRAINING_OUTPUT, exist_ok=True)
    os.makedirs(DRIVE_BEST, exist_ok=True)

    dataset_config, train_samples, eval_samples = load_dataset_config()

    # Auto-backup
    stop_backup = threading.Event()
    def backup_best():
        while not stop_backup.is_set():
            time.sleep(600)
            try:
                runs = sorted(glob.glob(os.path.join(TRAINING_OUTPUT, "*/")))
                if not runs: continue
                bests = sorted(glob.glob(os.path.join(runs[-1], "best_model_*.pth")),
                               key=lambda x: int(x.split("best_model_")[-1].replace(".pth","")))
                if not bests or os.path.getsize(bests[-1]) < 1e6: continue
                for old in glob.glob(os.path.join(DRIVE_BEST, "best_model_*.pth")): os.remove(old)
                shutil.copy(bests[-1], DRIVE_BEST)
                shutil.copy(os.path.join(runs[-1], "config.json"), DRIVE_BEST)
                print(f"\n💾 Backup: {os.path.basename(bests[-1])}")
            except Exception as e: print(f"\n⚠️ {e}")
    threading.Thread(target=backup_best, daemon=True).start()

    model_args = GPTArgs(
        max_conditioning_length=132300, min_conditioning_length=66150,
        debug_loading_failures=False, max_wav_length=255995, max_text_length=200,
        mel_norm_file=os.path.join(MODEL_DIR, "mel_stats.pth"),
        dvae_checkpoint=os.path.join(MODEL_DIR, "dvae.pth"),
        xtts_checkpoint=os.path.join(MODEL_DIR, "model.pth"),
        tokenizer_file=os.path.join(MODEL_DIR, "vocab.json"),
        gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
    )
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    config = GPTTrainerConfig(
        epochs=5, output_path=TRAINING_OUTPUT,
        model_args=model_args, run_name="xtts_darija_m1",
        audio=audio_config, batch_size=2, batch_group_size=48,
        eval_batch_size=2, num_loader_workers=0,
        save_step=1000, save_n_checkpoints=1, save_checkpoints=True,
        optimizer="AdamW", optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-6, lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        mixed_precision=True, test_sentences=[], datasets=[dataset_config],
    )

    model = GPTTrainer.init_from_config(config)
    gc.collect(); torch.cuda.empty_cache()

    trainer = Trainer(
        TrainerArgs(restore_path=restore_path, skip_train_epoch=False,
                    start_with_eval=False, grad_accum_steps=126),
        config, output_path=TRAINING_OUTPUT,
        model=model, train_samples=train_samples, eval_samples=eval_samples,
    )

    print(f"\n🚀 Fine-tuning démarré...")
    try:
        trainer.fit()
        print("✅ Fine-tuning terminé")
    finally:
        stop_backup.set()
        runs = sorted(glob.glob(os.path.join(TRAINING_OUTPUT, "*/")))
        if runs:
            bests = sorted(glob.glob(os.path.join(runs[-1], "best_model_*.pth")),
                           key=lambda x: int(x.split("best_model_")[-1].replace(".pth","")))
            if bests and os.path.getsize(bests[-1]) > 1e6:
                for old in glob.glob(os.path.join(DRIVE_BEST, "best_model_*.pth")): os.remove(old)
                shutil.copy(bests[-1], DRIVE_BEST)
                shutil.copy(os.path.join(runs[-1], "config.json"), DRIVE_BEST)
                print(f"✅ Backup final: {os.path.basename(bests[-1])}")


if __name__ == "__main__":
    setup_base_model()
    finetune()
