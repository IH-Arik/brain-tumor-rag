#!/usr/bin/env python3
"""
GitHub Deployment Script for Brain Tumor RAG System
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_git_repo():
    """Check if we're in a git repository"""
    if not os.path.exists('.git'):
        print("❌ Not a git repository. Initializing...")
        run_command("git init", "Initializing git repository")
        return False
    return True

def setup_gitignore():
    """Ensure .gitignore is properly set up"""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("❌ .gitignore not found. Creating...")
        return False
    
    # Check if essential patterns are in .gitignore
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    required_patterns = ['*.pth', 'uploads/', '*.faiss', '__pycache__']
    missing = [pattern for pattern in required_patterns if pattern not in content]
    
    if missing:
        print(f"⚠️  Missing patterns in .gitignore: {missing}")
        return False
    
    return True

def create_github_repo():
    """Create GitHub repository (manual step instructions)"""
    print("\n📋 Manual Steps Required:")
    print("1. Go to https://github.com and create a new repository")
    print("2. Name it 'brain-tumor-rag' (or your preferred name)")
    print("3. Don't initialize with README (we already have one)")
    print("4. Keep it private or public as you prefer")
    print("\nOnce created, GitHub will show you commands like:")
    print("git remote add origin https://github.com/yourusername/brain-tumor-rag.git")
    print("git branch -M main")
    print("git push -u origin main")

def deploy_to_github():
    """Main deployment function"""
    print("🚀 Starting GitHub Deployment for Brain Tumor RAG System")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Make sure you're in the project directory.")
        return False
    
    # Check git repository
    check_git_repo()
    
    # Setup .gitignore
    if not setup_gitignore():
        print("❌ .gitignore setup incomplete. Please check the file.")
        return False
    
    # Add all files
    run_command("git add .", "Adding files to git")
    
    # Check status
    status = run_command("git status --porcelain", "Checking git status")
    if status:
        print(f"📁 Files to be committed:\n{status}")
    
    # Commit changes
    commit_msg = "feat: Add brain tumor classification with RAG system"
    run_command(f'git commit -m "{commit_msg}"', "Committing changes")
    
    # Instructions for GitHub setup
    create_github_repo()
    
    print("\n🎯 Next Steps:")
    print("1. Create the GitHub repository as shown above")
    print("2. Run the git commands GitHub provides")
    print("3. Your code will be deployed to GitHub!")
    
    print("\n🌐 Deployment Options:")
    print("• GitHub Pages: For frontend only")
    print("• Railway: Easy full-stack deployment")
    print("• Heroku: Classic cloud platform")
    print("• Docker: Containerized deployment")
    
    return True

def show_deployment_commands():
    """Show commands for different deployment platforms"""
    print("\n📚 Deployment Commands:")
    print("=" * 40)
    
    print("\n🚂 Railway Deployment:")
    print("npm install -g @railway/cli")
    print("railway login")
    print("railway init")
    print("railway up")
    
    print("\n🔷 Heroku Deployment:")
    print("heroku create your-app-name")
    print("git push heroku main")
    
    print("\n🐳 Docker Deployment:")
    print("docker build -t brain-tumor-rag .")
    print("docker run -p 5000:5000 brain-tumor-rag")
    
    print("\n📦 Docker Compose:")
    print("docker-compose up -d")

if __name__ == "__main__":
    print("🧠 Brain Tumor RAG System - GitHub Deployment")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python deploy_github.py")
        print("This script helps you deploy your RAG system to GitHub")
        show_deployment_commands()
        sys.exit(0)
    
    success = deploy_to_github()
    
    if success:
        show_deployment_commands()
        print("\n✅ Deployment preparation complete!")
        print("🎉 Your Brain Tumor RAG System is ready for GitHub deployment!")
    else:
        print("\n❌ Deployment preparation failed. Please check the errors above.")
        sys.exit(1)
