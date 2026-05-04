import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Sparkles, Loader2 } from 'lucide-react'
import api from '../../lib/api'

export default function Orchestrator() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      // Get user info from localStorage
      const userStr = localStorage.getItem('user')
      const user = userStr ? JSON.parse(userStr) : null
      const userId = user?.id || user?._id || 'anonymous'

      const response = await api.post('/orchestrator/workflow', {
        operation: 'complex_workflow',
        parameters: {
          description: prompt,
          content_types: ['text'],
          quality_level: 'high'
        },
        user_id: userId,
        priority: 2
      })
      
      setResult(response.data.message || 'Workflow completed successfully.')
    } catch (error: any) {
      console.error('Workflow execution failed:', error)
      const errorMsg = error.response?.data?.detail?.message || error.response?.data?.detail || 'Failed to execute workflow. Please try again.'
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
          <h1 className="text-4xl font-bold">AI Orchestrator</h1>
          <p className="text-slate-400">Combine multiple AI engines into workflows</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Workflow Description</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Describe your workflow... (e.g., Create a blog post with images and optimize for social media)"
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
              <span>Executing Workflow...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Execute Workflow</span>
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Workflow Results</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <p className="whitespace-pre-wrap">{result}</p>
          </div>
        </div>
      )}
    </div>
  )
}
