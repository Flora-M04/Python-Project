#!/bin/bash

echo "🔁 Changing to project directory..."
cd /mnt/d/python/Group_D || { echo '❌ Directory not found'; exit 1; }

echo "📦 Creating virtual environment (venv)..."
python3 -m venv venv

echo "✅ Activating virtual environment..."
source venv/bin/activate

echo "⬇️ Installing required packages..."
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel

echo "🧠 Registering venv as Jupyter kernel..."
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

echo "🚀 Launching VS Code..."
code .

echo "✅ All done! In VS Code, open your .ipynb and select the 'Python (venv)' kernel."
