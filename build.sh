#!/bin/bash

# Définir la version de Python à utiliser
PYTHON_VERSION="python3.10"
PIP_VERSION="pip3.10"

echo "--- Démarrage du script de build personnalisé ---"
echo "Utilisation de la version de Python: $PYTHON_VERSION"

# Vérifier si les binaires Python et Pip existent
if ! command -v "$PYTHON_VERSION" &> /dev/null; then
    echo "Erreur: Python version $PYTHON_VERSION non trouvé. Veuillez vérifier les runtimes disponibles sur Render."
    exit 1
fi

if ! command -v "$PIP_VERSION" &> /dev/null; then
    echo "Erreur: $PIP_VERSION non trouvé. Veuillez vérifier les runtimes disponibles sur Render."
    exit 1
fi

# Installer les dépendances Python en utilisant la version spécifique de pip
echo "Installation des dépendances Python depuis requirements.txt avec $PIP_VERSION..."
"$PIP_VERSION" install --disable-pip-version-check --target . --upgrade -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Erreur lors de l'installation des dépendances Python."
    exit 1
fi

echo "Dépendances Python installées avec succès."

# Créer le répertoire de sortie si nécessaire (Render s'attend à ce que les fonctions soient dans un certain chemin)
# Note: Pour Render, le répertoire de sortie n'est pas toujours nécessaire si l'application est lancée directement.
# Mais si vous avez une configuration de build complexe, cela peut être utile.
# Pour une application FastAPI simple, Render s'attend à ce que le point d'entrée soit accessible.
# Nous allons copier les fichiers de l'API dans le répertoire racine du build pour s'assurer qu'ils sont trouvés.
mkdir -p api # S'assurer que le dossier api existe
cp -r api/* . # Copier le contenu de api/ vers la racine du build

echo "Fichiers de l'API copiés dans le répertoire de sortie."

echo "--- Script de build personnalisé terminé ---"
