#!/bin/bash
# Script to push to remote repository
# Usage: ./push_to_remote.sh <repository-url>

if [ -z "$1" ]; then
    echo "Usage: ./push_to_remote.sh <repository-url>"
    echo ""
    echo "Examples:"
    echo "  ./push_to_remote.sh https://github.com/USERNAME/livnium-quantum.git"
    echo "  ./push_to_remote.sh git@github.com:USERNAME/livnium-quantum.git"
    exit 1
fi

REPO_URL=$1

echo "Setting up remote repository..."
git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"

echo "Pushing to remote..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to remote!"
    echo "Repository: $REPO_URL"
    git remote -v
else
    echo ""
    echo "❌ Push failed. Check your repository URL and permissions."
fi

