import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store/authStore'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import Dashboard from './pages/Dashboard'
import ContentList from './pages/ContentList'
import ContentCreate from './pages/ContentCreate'
import ContentDiscovery from './pages/ContentDiscovery'
import ContentScheduler from './pages/ContentScheduler'
import Analytics from './pages/Analytics'
import AIEngines from './pages/AIEngines'
import Jobs from './pages/Jobs'
import Settings from './pages/Settings'
import Layout from './components/Layout'
import TextIntelligence from './pages/engines/TextIntelligence'
import ImageGeneration from './pages/engines/ImageGeneration'
import AudioGeneration from './pages/engines/AudioGeneration'
import VideoGeneration from './pages/engines/VideoGeneration'
import CreativeAssistant from './pages/engines/CreativeAssistant'
import SocialMediaPlanner from './pages/engines/SocialMediaPlanner'
import DiscoveryAnalytics from './pages/engines/DiscoveryAnalytics'
import Orchestrator from './pages/engines/Orchestrator'

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore()
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        
        <Route path="/app" element={
          <PrivateRoute>
            <Layout />
          </PrivateRoute>
        }>
          <Route index element={<Dashboard />} />
          <Route path="discover" element={<ContentDiscovery />} />
          <Route path="content" element={<ContentList />} />
          <Route path="content/create" element={<ContentCreate />} />
          <Route path="scheduler" element={<ContentScheduler />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="engines" element={<AIEngines />} />
          <Route path="engines/text" element={<TextIntelligence />} />
          <Route path="engines/image" element={<ImageGeneration />} />
          <Route path="engines/audio" element={<AudioGeneration />} />
          <Route path="engines/video" element={<VideoGeneration />} />
          <Route path="engines/creative" element={<CreativeAssistant />} />
          <Route path="engines/social" element={<SocialMediaPlanner />} />
          <Route path="engines/analytics" element={<DiscoveryAnalytics />} />
          <Route path="engines/orchestrator" element={<Orchestrator />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
