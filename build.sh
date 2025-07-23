#!/bin/bash

# Définir la version de Python à utiliser
# Vercel supporte généralement python3.9, python3.10, python3.11, python3.12
# Nous allons essayer python3.10 car c'est une version stable et largement compatible.
PYTHON_VERSION="python3.10"

echo "--- Démarrage du script de build personnalisé ---"
echo "Utilisation de la version de Python: $PYTHON_VERSION"

# Vérifier si le répertoire /usr/local/bin existe et est dans le PATH
if [[ ":$PATH:" != *":/usr/local/bin:"* ]]; then
  export PATH="/usr/local/bin:$PATH"
fi

# Installer pyenv si ce n'est pas déjà fait (pour gérer les versions de Python)
# Note: Vercel a déjà des runtimes Python pré-installés, mais cette approche est plus robuste
# si la détection automatique échoue.
# Cependant, pour Vercel, il est souvent plus simple d'utiliser les binaires directement.
# Nous allons simplifier et utiliser les binaires Vercel si disponibles.

# Assurez-vous que pip est disponible pour la version de Python souhaitée
# Vercel fournit généralement des binaires comme python3.10 et pip3.10
if command -v "$PYTHON_VERSION" &> /dev/null; then
    echo "Python version $PYTHON_VERSION trouvé."
else
    echo "Erreur: Python version $PYTHON_VERSION non trouvé. Veuillez vérifier les runtimes disponibles sur Vercel."
    exit 1
fi

if command -v "pip3.10" &> /dev/null; then
    echo "pip3.10 trouvé."
else
    echo "Erreur: pip3.10 non trouvé. Veuillez vérifier les runtimes disponibles sur Vercel."
    exit 1
fi

# Installer les dépendances Python
echo "Installation des dépendances Python depuis requirements.txt..."
pip3.10 install --disable-pip-version-check --target . --upgrade -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Erreur lors de l'installation des dépendances Python."
    exit 1
fi

echo "Dépendances Python installées avec succès."

# Créer le répertoire de sortie si nécessaire (Vercel s'attend à ce que les fonctions soient dans un certain chemin)
mkdir -p .vercel/output/functions/api

# Copier les fichiers de l'API dans le répertoire de sortie attendu par Vercel
# Assurez-vous que votre structure de fichiers est api/main.py et api/app/chatbot.py
cp -r api/* .vercel/output/functions/api/

echo "Fichiers de l'API copiés dans le répertoire de sortie."

echo "--- Script de build personnalisé terminé ---"
