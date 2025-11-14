#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide the path to the custom_nodes directory."
    exit 1
fi

# Use the provided directory
custom_nodes_dir="$1"

# Change to the custom_nodes directory
cd "$custom_nodes_dir" || exit 1

echo "Processing custom nodes in: $custom_nodes_dir"

# Loop through immediate subdirectories
for dir in */; do
    if [ -d "$dir" ]; then
        # Remove the trailing slash
        dir_name="${dir%/}"

        # Check if the directory name ends with ".disabled"
        if [[ "$dir_name" == *.disabled ]]; then
            echo "Skipping directory (ends with .disabled): $dir_name"
            continue
        fi

        echo "Processing directory: $dir_name"

        # Change to the subdirectory
        cd "$dir" || continue

        # Perform git pull if it's a git repository
        if [ -d .git ]; then
            echo "Performing git pull in $dir_name"
            git pull
        fi

        # Check if requirements.txt exists and install if it does
        if [ -f requirements.txt ]; then
            echo "Installing requirements in $dir_name"
            pip install -r requirements.txt
        fi

        # Return to the custom_nodes directory
        cd "$custom_nodes_dir"
    fi
done

echo "All operations completed."
