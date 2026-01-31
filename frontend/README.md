# ContentFlow AI - Frontend

Modern, glassmorphic React + TypeScript frontend for ContentFlow AI.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/          # Page components
│   ├── store/          # Zustand state management
│   ├── lib/            # Utilities and API client
│   ├── App.tsx         # Main app component
│   ├── main.tsx        # Entry point
│   └── index.css       # Global styles
├── public/             # Static assets
└── package.json        # Dependencies
```

## 🎨 Design System

### Colors
- **Primary**: Indigo/Purple gradient (#6366f1 → #8b5cf6)
- **Accent**: Amber/Pink gradient (#f59e0b → #ec4899)
- **Background**: Dark slate (#0f172a, #1e293b)

### Features
- ✨ Glassmorphism design
- 🎭 Smooth animations with Framer Motion
- 🌙 Dark mode native
- 📱 Fully responsive
- ♿ Accessible components

## 🛠️ Tech Stack

- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **State**: Zustand
- **Forms**: React Hook Form + Zod
- **Icons**: Lucide React
- **Charts**: Recharts
- **HTTP**: Axios

## 📦 Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

## 🔗 API Integration

The frontend connects to the backend API at `http://localhost:8000/api/v1`

Proxy configuration in `vite.config.ts` handles CORS during development.

## 🎯 Features

- 🔐 Authentication (Login/Register)
- 📊 Dashboard with analytics
- 📝 Content management (CRUD)
- 🤖 7 AI engines integration
- 📈 Job tracking and progress
- ⚙️ User settings
- 🎨 Modern glassmorphic UI

## 🚧 Development Status

✅ Project structure
✅ Landing page
✅ Authentication flow
🚧 Dashboard (in progress)
🚧 Content management (in progress)
🚧 AI engines interface (in progress)

## 📝 Notes

- Make sure the backend is running before starting the frontend
- The app uses JWT tokens stored in localStorage
- All API calls go through the axios instance in `src/lib/api.ts`

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details
