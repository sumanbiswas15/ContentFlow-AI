import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Sparkles, Loader2 } from 'lucide-react'
import api from '../../lib/api'

export default function TextIntelligence() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    setError('')
    setResult('')
    
    try {
      console.log('Sending request to /engines/text/generate with:', {
        content_type: 'blog',
        prompt: prompt,
        tone: 'professional',
        language: 'en'
      })
      
      const response = await api.post('/engines/text/generate', {
        content_type: 'blog',
        prompt: prompt,
        tone: 'professional',
        language: 'en'
      })
      
      console.log('Response received:', response.data)
      
      if (response.data.content) {
        setResult(response.data.content)
      } else {
        setError('No content received from the server. Response: ' + JSON.stringify(response.data))
      }
    } catch (error: unknown) {
      console.error('Generation failed:', error)
      const err = error as { 
        response?: { 
          status?: number
          data?: { detail?: string | { message?: string } } 
        }
        message?: string
      }
      
      // Handle different error formats
      let errorMsg = 'Failed to generate content. '
      
      if (err.response) {
        errorMsg += `Status: ${err.response.status}. `
        if (err.response.data?.detail) {
          if (typeof err.response.data.detail === 'string') {
            errorMsg += err.response.data.detail
          } else if (err.response.data.detail.message) {
            errorMsg += err.response.data.detail.message
          }
        }
      } else if (err.message) {
        errorMsg += err.message
      }
      
      console.error('Error details:', errorMsg)
      setError(errorMsg)
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
          <h1 className="text-4xl font-bold">Text Intelligence</h1>
          <p className="text-slate-400">Generate and transform text with AI</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Your Prompt</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Enter your prompt here... (e.g., Write a blog post about AI)"
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
              <Sparkles className="w-5 h-5" />
              <span>Generate</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="glass rounded-xl p-6 space-y-4 border-2 border-red-500">
          <h2 className="text-xl font-semibold text-red-500">Error</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-red-700">
            <p className="whitespace-pre-wrap text-red-300">{error}</p>
          </div>
          <p className="text-sm text-slate-400">Check the browser console (F12) for more details.</p>
        </div>
      )}

      {result && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Generated Content</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <p className="whitespace-pre-wrap">{result}</p>
          </div>
        </div>
      )}
    </div>
  )
}
