# Projet INF581 - Solving a Rubik's Cube

## Setup Windows
- Créer un venv avec `python -m venv venv`
- Entrer dans le venv avec `./venv/Scripts/activate`
- Installer les dependencies avec `pip install -r requirements.txt`

Si erreur de Windows qui indique que l'éxécution de scripts n'est pas activée sur le système, lancer PowerShell en mode administrateur et faire la commande `set-executionpolicy unrestricted`.

### Commentaires
- jeu où toutes les infos sont dispos
- state space enormissime pour le 3x3 (43.10^18)
- 2x2 beaucoup plus simple car seulement 3M de combis
- tout est wrappé dans un conteneur gym