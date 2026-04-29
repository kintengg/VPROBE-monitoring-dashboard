# v0-build-surveillance-system

This is a [Next.js](https://nextjs.org) project bootstrapped with [v0](https://v0.app).

## Built with v0

This repository is linked to a [v0](https://v0.app) project. You can continue developing by visiting the link below -- start new chats to make changes, and v0 will push commits directly to this repo. Every merge to `main` will automatically deploy.

[Continue working on v0 →](https://v0.app/chat/projects/prj_nsVMvzEhSjJ8KFbiyZfOh0G4NSiB)

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

## Learn More

To learn more, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
- [v0 Documentation](https://v0.app/docs) - learn about v0 and how to use it.

## Backend Setup (Python/FastAPI)

Use these steps to run the API locally from the repo root:

```bash
# Create a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Create a virtual environment (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate
```

```bash
# Install backend requirements
pip install -r requirements.txt

# Install FastAPI extras used by the API
pip install "fastapi[standard]" python-multipart
```

```bash
# Start the API server (from repo root)
uvicorn backend.app.main:app --reload
```

## Frontend Setup (Node/Next.js)

Use these steps to run the dashboard locally from the repo root:

```bash
# Install Node dependencies
npm install
```

```bash
# Start the frontend dev server
npm run dev
```

<a href="https://v0.app/chat/api/kiro/clone/CornOnTheKob/v0-build-surveillance-system" alt="Open in Kiro"><img src="https://pdgvvgmkdvyeydso.public.blob.vercel-storage.com/open%20in%20kiro.svg?sanitize=true" /></a>
