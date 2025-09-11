# Backend
cd D:\project-music\project-music\backend
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# (ถ้าไม่มี reqs ให้ติดตั้งชุดพื้นฐาน + openunmix ตามด้านบน)
cd ..
uvicorn backend.main:app --reload --port 8000

# Frontend
cd D:\project-music\project-music\frontend
npm install
npm run dev
