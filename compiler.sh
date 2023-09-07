#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

python3 -m venv easystreamEnv
source ./easystreamEnv/bin/activate

python3 -m pip install -U pip
python3 -m pip install -r ./requirements_linux.txt
python3 -m pip install pyinstaller

# Spécifiez le chemin du dossier source contenant les fichiers Python
dossier_source="./src/"
# Spécifiez le chemin du dossier de destination (racine)
dossier_destination="./build/"

# Assurez-vous que le dossier de destination existe, sinon créez-le
mkdir -p "$dossier_destination"
rm -rf "$dossier_destination/*"

# Move python file in project to a build directory
find "$dossier_source" -type f -name "*.py" -exec cp {} "$dossier_destination" \;
# Replace lines like `from src.dir.a import a` with `from a import a`
find "$dossier_destination" -type f -name "*.py" -exec sed -i 's/^from src\.[^.]*\./from /' {} \;
# Replace lines like `from src.a import a` with `from a import a`
find "$dossier_destination" -type f -name "*.py" -exec sed -i 's/^from src\./from /' {} \;

cd "$dossier_destination"
pyinstaller ./main.py --paths=.\easystreamEnv\Lib\site-packages\ --onefile -n echostra
cd ..