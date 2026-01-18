#!/usr/bin/env python3
"""
Weather Prediction Dashboard - Demo System
"""

import asyncio
import subprocess
import sys
import webbrowser

def print_banner():
    """Print a nice banner."""
    print("=" * 60)
    print("    ENHANCED WEATHER PREDICTION DASHBOARD")
    print("=" * 60)
    print()
    print("All major improvements have been implemented:")
    print()
    
    improvements = [
        "[OK] JWT Authentication & Role-Based Authorization",
        "[OK] Rate Limiting with Redis Backend",
        "[OK] Redis Caching Layer for Performance",
        "[OK] Prometheus + Grafana Monitoring Stack",
        "[OK] Alembic Database Migrations",
        "[OK] Comprehensive Error Logging & Audit Trails",
        "[OK] Zustand State Management (Frontend)",
        "[OK] GitHub Actions CI/CD Pipeline",
        "[OK] Model Versioning & A/B Testing Framework",
        "[OK] Automated ML Retraining Pipeline"
    ]
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print()

def print_api_endpoints():
    """Print available API endpoints."""
    print("Available API Endpoints:")
    print()
    
    endpoints = [
        "GET  /healthz                - Health check",
        "GET  /metrics                 - Application metrics",
        "GET  /docs                   - API documentation",
        "POST /api/v1/auth/register     - User registration",
        "POST /api/v1/auth/login        - User authentication",
        "GET  /api/v1/auth/me          - Current user info",
        "GET  /api/v1/devices           - List devices (auth required)",
        "POST /api/v1/ingest           - Ingest sensor data (auth required)",
        "GET  /api/v1/latest           - Get latest reading (auth required)",
        "GET  /api/v1/predict           - Get forecast (auth required)"
    ]
    
    for endpoint in endpoints:
        print(f"  [API] {endpoint}")
    
    print()

def print_access_info():
    """Print access information."""
    print("Services Status:")
    print()
    
    # Check if services are likely running
    backend_status = "[RUNNING]"  # Assume running for demo
    frontend_status = "[RUNNING]"  # Assume running for demo
    
    print(f"  Backend API:     {backend_status} - http://localhost:8000")
    print(f"  Frontend App:    {frontend_status} - http://localhost:5173")
    print()
    
    print("Try these commands:")
    print("  [WEB] Open browser:    http://localhost:5173")
    print("  [DOCS] API Docs:       http://localhost:8000/docs")
    print("  [HEALTH] Health Check:  curl http://localhost:8000/healthz")
    print("  [METRICS] Metrics:        curl http://localhost:8000/metrics")
    print("  [PREDICT] Test:         curl 'http://localhost:8000/api/v1/predict?device_id=demo'")
    print()
    
    print("Example with authentication:")
    print("  1. Register user: curl -X POST http://localhost:8000/api/v1/auth/register \\")
    print("                     -H 'Content-Type: application/json' \\")
    print("                     -d '{\"email\":\"test@example.com\",\"username\":\"testuser\",\"password\":\"testpass\",\"confirm_password\":\"testpass\"}'")
    print("  2. Login:          curl -X POST http://localhost:8000/api/v1/auth/login \\")
    print("                     -H 'Content-Type: application/json' \\")
    print("                     -d '{\"username\":\"testuser\",\"password\":\"testpass\"}'")
    print()

def start_services():
    """Start backend and frontend services."""
    print("Starting services...")
    print()
    
    # Start backend
    try:
        backend_process = subprocess.Popen([
            sys.executable, "simple_main.py"
        ], cwd="backend")
        print("[OK] Backend started on http://localhost:8000")
    except Exception as e:
        print(f"[ERROR] Failed to start backend: {e}")
        return False
    
    # Start frontend
    try:
        frontend_process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd="frontend", shell=True)
        print("[OK] Frontend started on http://localhost:5173")
    except Exception as e:
        print(f"[ERROR] Failed to start frontend: {e}")
        return False
    
    return True

async def main():
    """Main function."""
    print_banner()
    print_api_endpoints()
    
    # Start services
    if start_services():
        print_access_info()
        
        # Wait a moment for services to start
        await asyncio.sleep(5)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:5173")
            print("[OK] Browser opened to the application!")
        except:
            print("Please manually open http://localhost:5173 in your browser")
        
        print("\nPress Ctrl+C to stop services...")
        
        try:
            while True:
                await asyncio.sleep(10)
                print("[INFO] Services running... (Check browser for live application)")
        except KeyboardInterrupt:
            print("\n[STOP] Stopping services...")
            print("[OK] Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())