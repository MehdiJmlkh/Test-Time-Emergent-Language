#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RESET='\033[0m'

echo "Choose mode:"
echo "1) baseline"
echo "2) VQEL"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    NOTEBOOK_NAME="src/0_baseline_reinforce.ipynb"
    echo -e "${GREEN}Running BASELINE${RESET}"
elif [ "$choice" == "2" ]; then
    NOTEBOOK_NAME="src/0_vq_method.ipynb"
    echo -e "${GREEN}Running VQ-EL${RESET}"
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# NOTEBOOK_NAME="src/0_vq_method.ipynb"
# NOTEBOOK_NAME="src/0_baseline_reinforce.ipynb"
RUN_DIR="src/runs"
OUT_DIR="output"
CONFIG_DIR="configs"
EXP_DIR="experiments"

if [ -d "$CONFIG_DIR" ]; then
    rm -rf "$CONFIG_DIR"
fi

mkdir -p "$CONFIG_DIR"
mkdir -p "$EXP_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "$OUT_DIR"


python3 setup.py


for config_file in $(ls "$CONFIG_DIR"/config*.py | sort -V); do
    config_name=$(basename "$config_file" .py)
    output_file="../$OUT_DIR/output.ipynb"
    
    echo -e "${BLUE}Running notebook with config: $config_file${RESET}"

    CONFIG="../$config_file" jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.allow_errors=True \
        --output="$output_file" \
        "$NOTEBOOK_NAME"
    
    
    # Move last run to experiments directory
    LAST_EXP=$(ls -1 "$RUN_DIR" | sort | tail -n 1)
    LAST_RUN="$RUN_DIR/$LAST_EXP"

    if [ -d "$LAST_RUN" ]; then
        mv "$LAST_RUN" "$EXP_DIR/"
    else
        echo "Error: $LAST_RUN is not a directory."
        exit 1
    fi

    # Move output file to the last experiment directory
    cp -r "$OUT_DIR" "$EXP_DIR/$LAST_EXP"

    echo -e "${GREEN}Saved output to: $output_file${RESET}"
done

rm -rf "$CONFIG_DIR"
rm -rf "$OUT_DIR"
