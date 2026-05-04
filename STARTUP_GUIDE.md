# ContentFlow AI - Startup Guide

## Prerequisites
- Python 3.10+
- Node.js 18+
- Docker Desktop (for MongoDB and Redis)

## Quick Start

### 1. Start Database Services
```powershell
docker-compose up -d mongodb redis
```

### 2. Start Backend Server
```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Start Frontend (in a new terminal)
```powershell
cd frontend
npm run dev
```

## Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## User Credentials
- **Username**: suman112
- **Email**: rksb1507@gmail.com
- **Password**: (your password)

## Troubleshooting

### 401 Unauthorized Errors
If you see 401 errors after login, the JWT middleware might not be working. The app will still function but some features may be limited.

### Backend Won't Start
- Check if MongoDB and Redis are running: `docker-compose ps`
- Check if port 8000 is available: `netstat -ano | findstr :8000`

### Frontend Won't Start
- Check if port 3000 is available
- Try `npm install` in the frontend directory

## Features
- ✅ User Registration & Login
- ✅ 7 AI Engines (Text, Image, Audio, Video, Social Media, Analytics, Creative Assistant)
- ✅ Content Management
- ✅ Job Processing
- ✅ Dashboard with Stats
