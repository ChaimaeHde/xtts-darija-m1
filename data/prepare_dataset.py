"""
Préparation du dataset DODa pour le fine-tuning XTTS-v2
Locuteur M1 : indices 4000-4649 (≈ 35 minutes)
"""

import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import subprocess

def download_and_prepare(
    output_dir="doda_m1_35min",
    start_idx=4000,
    end_idx=4650,
    text_col="darija_Arab_new"
):
    """Télécharge, prépare et convertit les données DODa pour M1."""

    print(f"⏳ Chargement du dataset DODa...")
    ds = load_dataset("atlasia/DODa-audio-dataset", split="train")
    subset = ds.select(range(start_idx, end_idx))
    print(f"✅ {len(subset)} samples sélectionnés (indices {start_idx}-{end_idx})")

    # Créer dossiers
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # Sauvegarder les WAV
    rows = []
    for i, item in enumerate(tqdm(subset, desc="Sauvegarde WAV")):
        audio_array = item["audio"]["array"]
        sample_rate = item["audio"]["sampling_rate"]
        file_name   = f"utt_{i:04d}.wav"
        audio_path  = os.path.join(wav_dir, file_name)
        sf.write(audio_path, audio_array, sample_rate)
        text = item.get(text_col, "")
        rows.append({"file_name": file_name, "text": text})

    # Sauvegarder metadata
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"✅ {len(rows)} samples sauvegardés dans {output_dir}/")

    # Nettoyage texte
    df = pd.read_csv(os.path.join(output_dir, "metadata.csv"))
    def clean_text(text):
        if not isinstance(text, str): return None
        text = text.strip()
        return text if len(text) >= 3 else None

    df["text_norm"] = df["text"].apply(clean_text)
    df_clean = df.dropna(subset=["text_norm"])
    df_clean = df_clean[df_clean["text_norm"].str.len() > 3]
    print(f"✅ {len(df_clean)} samples après nettoyage (supprimé {len(df)-len(df_clean)})")

    # Créer train.csv format LJSpeech
    train_csv = os.path.join(output_dir, "train.csv")
    with open(train_csv, "w", encoding="utf-8") as f:
        for _, row in df_clean.iterrows():
            f.write(f"wavs/{row['file_name']}|{row['text_norm']}|{row['text_norm']}\n")

    # Fix IDs (supprimer wavs/ prefix et .wav extension)
    df_csv = pd.read_csv(train_csv, sep="|", header=None, names=["id", "text1", "text2"])
    df_csv["id"] = df_csv["id"].str.replace("wavs/", "", regex=False).str.replace(".wav", "", regex=False)
    df_csv.to_csv(train_csv, sep="|", header=False, index=False)
    print(f"✅ train.csv créé : {len(df_csv)} lignes")

    # Conversion 22050 Hz
    print("🔄 Conversion WAV à 22050 Hz...")
    wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
    import soundfile as sf as sf_check
    sample_info = sf_check.info(os.path.join(wav_dir, wavs[0]))

    if sample_info.samplerate != 22050:
        errors = []
        for fname in tqdm(wavs, desc="Conversion"):
            src = os.path.join(wav_dir, fname)
            tmp = src.replace(".wav", "_tmp.wav")
            ret = subprocess.run(
                ["ffmpeg", "-y", "-i", src, "-ar", "22050", "-ac", "1", "-sample_fmt", "s16", tmp],
                capture_output=True
            )
            if ret.returncode == 0:
                os.replace(tmp, src)
            else:
                errors.append(fname)
                if os.path.exists(tmp): os.remove(tmp)
        print(f"✅ Conversion terminée. Erreurs: {len(errors)}")
    else:
        print("✅ WAV déjà à 22050 Hz")

    return output_dir


if __name__ == "__main__":
    prepare_dataset = download_and_prepare()
    print(f"\n🎉 Dataset prêt dans : {prepare_dataset}")
