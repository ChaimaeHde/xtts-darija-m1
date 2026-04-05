"""
Point d'entrée principal — Interface Gradio TTS Darija M1
Usage : python app.py
        ou depuis Colab : exec(open("app.py").read())
"""

import sys
import os

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interface.gradio_app import demo

if __name__ == "__main__":
    print("🚀 Lancement de l'interface TTS Darija M1...")
    print("📦 Le modèle sera téléchargé automatiquement depuis HuggingFace")
    print("⏳ Premier lancement : ~2-3 minutes de téléchargement")
    demo.launch(
        share=True,    # Génère un lien public temporaire
        debug=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
