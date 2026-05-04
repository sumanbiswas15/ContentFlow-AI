import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Video, Loader2 } from 'lucide-react'
import api from '../../lib/api'

export default function VideoGeneration() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [videoUrl, setVideoUrl] = useState('')
  const [loading, setLoading] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      const response = await api.post('/engines/media/video/generate', {
        script: prompt,
        video_type: 'short_form',
        style: 'professional',
        specification: {
          width: 1920,
          height: 1080,
          format: 'mp4',
          quality: 'high',
          fps: 30,
          duration_seconds: 30,
          bitrate_kbps: 5000
        },
        include_audio: true,
        include_music: false,
        include_subtitles: false
      })
      setVideoUrl(response.data.file_url || '')
    } catch (error: any) {
      console.error('Generation failed:', error)
      const errorMsg = error.response?.data?.detail || 'Failed to generate video. Please try again.'
      alert(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center space-x-4">
        <button
          onClick={() => navigate('/app/engines')}
          className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
        <div>
          <h1 className="text-4xl font-bold">Video Pipeline</h1>
          <p className="text-slate-400">Create and edit videos with AI</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Video Description</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Describe the video you want to create... (e.g., A 30-second product explainer video)"
          />
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading || !prompt.trim()}
          className="w-full flex items-center justify-center space-x-2 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Video className="w-5 h-5" />
              <span>Generate Video</span>
            </>
          )}
        </button>
      </div>

      {videoUrl && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Generated Video</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <video controls className="w-full rounded-lg">
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support the video element.
            </video>
          </div>
        </div>
      )}
    </div>
  )
}
