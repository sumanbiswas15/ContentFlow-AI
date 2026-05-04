import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Music, Loader2 } from 'lucide-react'
import api from '../../lib/api'

export default function AudioGeneration() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [loading, setLoading] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      const response = await api.post('/engines/media/audio/generate', {
        text: prompt,
        audio_type: 'background_music',
        music_genre: 'ambient',
        mood: 'uplifting',
        tempo: 'medium',
        specification: {
          format: 'mp3',
          sample_rate: 44100,
          bitrate: 128,
          channels: 2,
          duration_seconds: 30
        }
      })
      setAudioUrl(response.data.file_url || '')
    } catch (error: unknown) {
      console.error('Generation failed:', error)
      const err = error as { response?: { data?: { detail?: string } } }
      const errorMsg = err.response?.data?.detail || 'Failed to generate audio. Please try again.'
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
          <h1 className="text-4xl font-bold">Audio Generation</h1>
          <p className="text-slate-400">Generate music, voiceovers, and narrations</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Audio Description</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Describe the audio you want to create... (e.g., Upbeat background music for a tech video)"
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
              <Music className="w-5 h-5" />
              <span>Generate Audio</span>
            </>
          )}
        </button>
      </div>

      {audioUrl && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Generated Audio</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <audio controls className="w-full">
              <source src={audioUrl} type="audio/mpeg" />
              Your browser does not support the audio element.
            </audio>
          </div>
        </div>
      )}
    </div>
  )
}
