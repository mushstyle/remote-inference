#!/bin/bash

# Check if API key is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <API_KEY>"
    echo "Example: $0 your_api_key_here"
    exit 1
fi

API_KEY=$1

# Check if .env exists, create if not
touch .env

# Remove any existing API_KEY line and add new one
# Using temporary file to handle inline editing reliably
grep -v '^API_KEY=' .env > .env.tmp
echo "API_KEY=$API_KEY" >> .env.tmp
mv .env.tmp .env

echo "API key has been set in .env file"