
ML.project.5th-sem — Combined Frontend + FastAPI Backend
=======================================================

Folder structure:
- frontend/    -> React + Vite + Tailwind (mock mode enabled)
- backend/     -> FastAPI backend (app.py, requirements.txt, Dockerfile)
- README.md    -> This file

Frontend expects POST /predict and GET /history at the backend (apiBase default 'http://localhost:8000').
The included FastAPI backend implements /predict and /history and will use model files (linear_reg_model.pkl and scaler.pkl) if they are present inside backend/.

How to run:
1) Frontend:
   cd frontend
   npm install
   npm run dev
   (open http://localhost:5173)

2) Backend:
   cd backend
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   uvicorn app:app --reload --host 0.0.0.0 --port 8000

Notes:
- Mock Mode in frontend is ON by default; to use real backend, turn Mock Mode OFF in the UI.
- If you want the backend to use the trained model, place linear_reg_model.pkl and scaler.pkl into the backend/ folder.

