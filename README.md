# 🏥 DIAGNOSTIX — AI-Driven Multi-Disease Detection Platform

> Harness the power of artificial intelligence for early disease detection. DIAGNOSTIX analyzes medical imaging and health data to deliver accurate diagnostic insights with explainable AI.

---

## 📌 Overview

**DIAGNOSTIX** is a full-stack web application that enables users to detect five major diseases using machine learning models. It combines a React + TypeScript frontend with a Python-based ML backend, Supabase for authentication and data storage, and explainability tools (Grad-CAM for images, SHAP for tabular data) to make AI predictions transparent and trustworthy.

---

## ✨ Features

- 🧠 **Brain Tumor Detection** — Analyzes MRI scans to identify potential brain tumors
- 🧬 **Alzheimer's Detection** — Detects early signs of Alzheimer's from brain MRI images
- 🫁 **Pneumonia Detection** — Classifies chest X-rays for signs of pneumonia
- 💉 **Diabetes Risk Prediction** — Predicts diabetes risk from health parameters (glucose, BMI, age, etc.)
- ❤️ **Heart Disease Risk Assessment** — Evaluates cardiovascular risk from clinical data
- 🤖 **AI Medical Chatbot** — Rule-based assistant for disease information and platform guidance
- 📊 **Results Dashboard** — View and track all past diagnostic results
- 🔐 **Authentication** — Secure sign-up/login via Supabase Auth
- 🔍 **Explainable AI** — Grad-CAM heatmaps for image models, SHAP values for tabular models

---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| React 18 + TypeScript | UI framework |
| Vite | Build tool & dev server |
| Tailwind CSS | Utility-first styling |
| shadcn/ui + Radix UI | Accessible component library |
| React Router v6 | Client-side routing |
| TanStack Query | Server state management |
| React Hook Form + Zod | Form handling & validation |
| Recharts | Data visualization |
| Supabase JS | Auth & database client |

### Backend
| Technology | Purpose |
|---|---|
| Python | ML model serving |
| Supabase | PostgreSQL database + Auth |

---

## 📁 Project Structure

```
diagnostix-hub/
├── public/                  # Static assets
├── src/
│   ├── components/
│   │   ├── ui/              # shadcn/ui components
│   │   ├── ChatbotModal.tsx # AI assistant modal
│   │   ├── DiseaseCard.tsx  # Disease selection card
│   │   ├── Navbar.tsx       # Top navigation bar
│   │   └── NavLink.tsx      # Navigation link component
│   ├── pages/
│   │   ├── Index.tsx        # Landing / home page
│   │   ├── Auth.tsx         # Login & registration
│   │   ├── PredictBrainTumor.tsx
│   │   ├── PredictAlzheimers.tsx
│   │   ├── PredictPneumonia.tsx
│   │   ├── PredictDiabetes.tsx
│   │   ├── PredictHeartDisease.tsx
│   │   └── Results.tsx      # Diagnostic history dashboard
│   ├── App.tsx              # Root component & routes
│   └── main.tsx             # App entry point
├── .env                     # Environment variables (never commit)
├── .env.example             # Template for environment variables
├── .gitignore
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── vite.config.ts
```

---

## 🚀 Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v18 or higher
- npm or bun
- A [Supabase](https://supabase.com/) project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/diagnostix-hub.git
cd diagnostix-hub
```

### 2. Install Dependencies

```bash
npm install
# or
bun install
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Fill in your Supabase credentials:

```env
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your_supabase_anon_key
VITE_SUPABASE_PROJECT_ID=your_project_id
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 4. Start the Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:8080`.

---

## 📜 Available Scripts

| Command | Description |
|---|---|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run build:dev` | Build in development mode |
| `npm run preview` | Preview production build locally |
| `npm run lint` | Run ESLint |

---

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `VITE_SUPABASE_URL` | Your Supabase project URL |
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Supabase anonymous/public API key |
| `VITE_SUPABASE_PROJECT_ID` | Your Supabase project ID |

---

## 🌐 Deployment

### Build for Production

```bash
npm run build
```

The output will be in the `dist/` folder. Deploy it to any static hosting provider:

- **Vercel** — Connect your GitHub repo and deploy instantly
- **Netlify** — Drag & drop the `dist/` folder or connect via Git
- **GitHub Pages** — Use the `gh-pages` package to publish the `dist/` folder

> Make sure to set your environment variables in the hosting platform's dashboard.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is private and not licensed for public distribution.

---

## 👤 Author
Harshit Rai

Built with ❤️ using [React](https://react.dev/), [Supabase](https://supabase.com/), and [shadcn/ui](https://ui.shadcn.com/).
