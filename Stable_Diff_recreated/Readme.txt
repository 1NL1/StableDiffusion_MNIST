Dans ce dossier, vous trouverez mon projet de recréation d'une IA par diffusion stable sans TP, et en ajoutant des aspects qui n'avaient pas été touchés comme le travail dans un espace latent, la prise en compte de prompts pour guider la génération des images ou la diffusion image à image.

Pour tester l'IA, vous pouvez utiliser pour les poids le fichier "v1-5-pruned-emaonly.ckpt" disponible sur hugginface à l'adresse https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main et à mettre dans le dossier "data"

Le dossier "model" contient tout les scripts en rapport avec le model, en particulier le fichier jupiter "demo.ipynb" pour lancer une démonstration
Le dossier "images" contient des images à utiliser dans la diffusion image à image
Le dossier "resultats" contient des exemples de résultats
Le dossier "data" contient trois fichiers pris sur hugginface (au lien précédent) et utiles pour les poids du modele, la lecture du prompt, etc 