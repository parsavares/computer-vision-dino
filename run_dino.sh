#!/bin/bash --login
#SBATCH --job-name=dino-spark
#SBATCH --output=run_%j.log
#SBATCH --time=33:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --partition=gpu



# Load required module (Iris)
module load lang/Python/3.11.5-GCCcore-13.2.0

# Environment setup
VENV_DIR="venv"
REQUIREMENTS="requirements.txt"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    
    if [ -f "$REQUIREMENTS" ]; then
        echo "Installing dependencies from $REQUIREMENTS..."
        pip install --upgrade pip
        pip install -r $REQUIREMENTS
    else
        echo "No requirements.txt found. Only creating virtual environment."
    fi
else
    echo "Activating existing virtual environment..."
    source $VENV_DIR/bin/activate
fi

PYTHON_SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

echo "=============================="
echo "STARTING PYTHON SCRIPT: $PYTHON_SCRIPT"
echo "ARGUMENTS: $SCRIPT_ARGS"
echo "=============================="

python "$PYTHON_SCRIPT" $SCRIPT_ARGS

