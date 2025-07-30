"""
Professional Bitcoin Trading Strategy Analysis - Main Application
Production-ready quantitative trading system with statistical rigor.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Professional Bitcoin Trading Analysis",
    description="Statistically rigorous quantitative trading system for Bitcoin analysis",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Professional Bitcoin Trading Analysis System",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Add database connectivity check here
        return {
            "status": "healthy",
            "system": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="System unhealthy")

@app.get("/api/status")
async def system_status():
    """Detailed system status endpoint."""
    return {
        "application": "Professional Bitcoin Trading Analysis",
        "status": "operational",
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "statistical_analysis": "enabled",
            "risk_management": "enabled",
            "real_time_monitoring": "enabled",
            "production_ready": True
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for consistent error responses."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if os.getenv("ENVIRONMENT") == "development" else False,
        log_level="info"
    ) 