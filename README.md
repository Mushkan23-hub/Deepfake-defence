# 🛡️ Deepfake Defence

Full-stack deepfake detection web app with AI + digital forensics.
Built by **Mushkan Bhagat** | VIT Bhopal University | Cyber Security & Digital Forensics

---

## What It Detects

| File Type | Detection Methods |
|---|---|
| **Image** | EfficientNet-B4 AI · Face morph · Noise analysis · ELA · ExifTool metadata |
| **Video** | Frame-by-frame AI · Face swap detection · ffprobe stream forensics · ExifTool |
| **Audio** | Voice clone (MFCC/pitch) · Voice morph · Splice detection · Edit detection |

## Features

- Login / Signup with hashed passwords (bcrypt)
- MongoDB scan history per user
- Dashboard: charts, threat timeline, stats by file type
- PDF forensic report download for every scan
- Fully responsive (mobile + desktop)
- Deployable on Railway (free tier)

---

## Local Setup (Windows)

### 1. Install prerequisites

- Python 3.11: https://www.python.org/downloads/
- MongoDB Community: https://www.mongodb.com/try/download/community
- ExifTool: https://exiftool.org → rename to exiftool.exe → put in C:\Windows\
- ffmpeg/ffprobe: https://ffmpeg.org/download.html → extract → put ffprobe.exe in C:\Windows\

### 2. Clone / create project folder

```
cd C:\Users\Mushkan\Desktop
mkdir deepfake-defence
cd deepfake-defence
```

Create all files as shown in the full guide.

### 3. Virtual environment

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Create .env file

```
SECRET_KEY=your_random_secret_here
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=deepfake_defence
```

### 5. Start MongoDB (separate CMD window)

```
mongod
```

### 6. Run the app

```
python app.py
```

Open: http://127.0.0.1:5000

---

## Deploy to Railway (Public URL)

### 1. MongoDB Atlas (free cloud database)

1. Go to https://cloud.mongodb.com → create free account
2. Create a free M0 cluster
3. Database Access → Add user (username + password)
4. Network Access → Allow from anywhere (0.0.0.0/0)
5. Connect → Drivers → copy the connection string
   Looks like: mongodb+srv://user:pass@cluster.mongodb.net/

### 2. GitHub

```
git init
git add .
git commit -m "Initial commit"
```

Create repo on GitHub, then:
```
git remote add origin https://github.com/YOUR_USERNAME/deepfake-defence.git
git push -u origin main
```

### 3. Railway

1. Go to https://railway.app → Login with GitHub
2. New Project → Deploy from GitHub repo → Select your repo
3. Add environment variables (Settings → Variables):
   - SECRET_KEY = any long random string
   - MONGO_URI  = your Atlas connection string
   - MONGO_DB   = deepfake_defence
4. Railway auto-detects Dockerfile and deploys
5. Settings → Networking → Generate Domain → your public URL!

---

## Project Structure

```
deepfake-defence/
├── app.py              ← Flask server + all routes
├── detector.py         ← Image + Video AI (EfficientNet-B4)
├── audio_detector.py   ← Voice clone + audio edit detection
├── forensics.py        ← ExifTool, ELA, ffprobe
├── database.py         ← MongoDB (users, scans, stats, timeline)
├── pdf_report.py       ← PDF forensic report generator
├── requirements.txt
├── Dockerfile          ← Railway deployment
├── railway.json
├── .env                ← secrets (never commit this)
├── templates/
│   ├── index.html      ← Login / Signup
│   ├── dashboard.html  ← Analytics + charts + threat timeline
│   └── analyse.html    ← Upload + full results
└── static/
    └── style.css       ← Fully responsive CSS
```
