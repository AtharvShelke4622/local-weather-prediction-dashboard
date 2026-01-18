#!/usr/bin/env python3
"""
Zero-Cost Weather Prediction Dashboard Deployment
"""

import subprocess
import sys
import os

def print_banner():
    print("=" * 60)
    print("    ZERO-COST WEATHER DASHBOARD DEPLOYMENT")
    print("=" * 60)
    print()

def check_prerequisites():
    print("Checking deployment prerequisites...")
    
    checks = []
    
    # Check Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(f"[OK] Node.js: {result.stdout.strip()}")
        else:
            checks.append("[FAIL] Node.js: Not installed")
    except:
        checks.append("[FAIL] Node.js: Not installed")
    
    # Check npm
    try:
        result = subprocess.run(["npm.cmd", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(f"[OK] npm: {result.stdout.strip()}")
        else:
            checks.append("[FAIL] npm: Not installed")
    except:
        checks.append("[FAIL] npm: Not installed")
    
    # Check Python
    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            checks.append(f"[OK] Python: {result.stdout.strip()}")
        else:
            checks.append("[FAIL] Python: Not installed")
    except:
        checks.append("[FAIL] Python: Not installed")
    
    # Check frontend build
    if os.path.exists("frontend/dist"):
        checks.append("[OK] Frontend: Built")
    else:
        checks.append("[WARN] Frontend: Not built (run npm run build)")
    
    # Check backend
    if os.path.exists("backend/main.py"):
        checks.append("[OK] Backend: Ready")
    else:
        checks.append("[FAIL] Backend: Not found")
    
    print("\n".join(checks))
    print()

def deploy_railway():
    print("Deploying backend to Railway...")
    
    try:
        result = subprocess.run(["railway", "deploy"], capture_output=True, text=True, cwd="backend")
        if result.returncode == 0:
            print("Backend deployed to Railway!")
            print(f"URL: {result.stdout.strip()}")
            return True
        else:
            print(f"Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def deploy_vercel():
    print("Deploying to Vercel...")
    
    try:
        result = subprocess.run(["npx", "vercel", "--prod", "frontend"], capture_output=True, text=True)
        if result.returncode == 0:
            print("Frontend deployed to Vercel!")
            print(f"URL: {result.stdout.strip()}")
            return True
        else:
            print(f"Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def deploy_github_pages():
    print("Deploying frontend to GitHub Pages...")
    
    try:
        # Build frontend
        result = subprocess.run("cd frontend && npm run build", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False
        
        # Deploy to GitHub Pages
        result = subprocess.run("cd frontend && vercel --prod", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Frontend deployed to GitHub Pages!")
            return True
        else:
            print(f"Deploy failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def setup_domain(domain):
    print(f"Setting up custom domain: {domain}")
    
    try:
        result = subprocess.run(["vercel", "domains", "add", domain], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Domain {domain} configured!")
            return True
        else:
            print(f"Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print_banner()
    
    command = sys.argv[1] if len(sys.argv) > 1 else "vercel"
    
    if command == "check":
        check_prerequisites()
    elif command == "railway":
        deploy_railway()
    elif command == "vercel":
        deploy_vercel()
    elif command == "github":
        deploy_github_pages()
    elif command == "domain" and len(sys.argv) > 2:
        setup_domain(sys.argv[2])
    else:
        print("Available commands:")
        print("  check     - Check deployment prerequisites")
        print("  railway   - Deploy backend to Railway")
        print("  vercel    - Deploy frontend to Vercel")
        print("  github    - Deploy frontend to GitHub Pages")
        print("  domain    - Setup custom domain")
        print()
        print("Usage: python deploy.py [command] [options]")

if __name__ == "__main__":
    main()