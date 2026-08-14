# Oncora Frontend

React (Vite) frontend for Oncora. Talks to the FastAPI backend in `../backend`.

## Running locally

```bash
npm install
npm run dev
```

Runs at http://localhost:5173 by default and expects the backend at
http://127.0.0.1:8000 (override with `VITE_API_URL` - see `.env.example`).
Each page falls back to mock data with a warning banner if the backend
isn't reachable, so the UI can be developed independently of it.

## Build

```bash
npm run build
```
