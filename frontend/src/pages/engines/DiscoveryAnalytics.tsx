import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Zap, Loader2 } from 'lucide-react'
import api from '../../lib/api'

export default function DiscoveryAnalytics() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      const response = await api.post('/engines/analytics/improvement-suggestions', {
        content: prompt,
        content_type: 'general'
      })
      
      // Format the suggestions into readable text
      const suggestions = response.data.suggestions || []
      const formattedResult = suggestions.map((s: { title: string; description: string; priority: string; expected_impact: string }, i: number) => 
        `${i + 1}. ${s.title}\n   ${s.description}\n   Priority: ${s.priority}\n   Impact: ${s.expected_impact}`
      ).join('\n\n')
      
      setResult(formattedResult || 'No suggestions available.')
    } catch (error: unknown) {
      console.error('Analysis failed:', error)
      const err = error as { response?: { data?: { detail?: string } } }
      const errorMsg = err.response?.data?.detail || 'Failed to analyze content. Please try again.'
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
          <h1 className="text-4xl font-bold">Discovery Analytics</h1>
          <p className="text-slate-400">Analyze and improve your content</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Content to Analyze</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Paste your content for analysis... (e.g., Blog post, video script, social post)"
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
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Zap className="w-5 h-5" />
              <span>Analyze Content</span>
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Analysis Results</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <p className="whitespace-pre-wrap">{result}</p>
          </div>
        </div>
      )}
    </div>
  )
}
