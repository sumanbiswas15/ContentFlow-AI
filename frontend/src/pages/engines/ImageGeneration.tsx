import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Image as ImageIcon, Loader2, Download, Upload } from 'lucide-react'
import api from '../../lib/api'

export default function ImageGeneration() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [imageUrl, setImageUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      const response = await api.post('/engines/media/image/generate', {
        prompt: prompt,
        image_type: 'thumbnail',
        style: 'professional',
        specification: {
          width: 1024,
          height: 1024,
          format: 'png',
          quality: 85
        }
      })
      setImageUrl(response.data.file_url || '')
    } catch (error: unknown) {
      console.error('Generation failed:', error)
      const err = error as { response?: { data?: { detail?: string } } }
      const errorMsg = err.response?.data?.detail || 'Failed to generate image. Please try again.'
      alert(errorMsg)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
    if (!imageUrl) return

    try {
      // Fetch the image
      const response = await fetch(`http://localhost:8000${imageUrl}`)
      const blob = await response.blob()
      
      // Create download link
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `generated-image-${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Download failed:', error)
      alert('Failed to download image')
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file')
      return
    }

    // Create object URL for preview
    const objectUrl = URL.createObjectURL(file)
    setImageUrl(objectUrl)
    
    // Optionally, you could upload to server here
    // For now, just showing the preview
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
          <h1 className="text-4xl font-bold">Image Generation</h1>
          <p className="text-slate-400">Create stunning visuals with AI</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-4">
          <p className="text-yellow-200 text-sm">
            <strong>Note:</strong> AI image generation requires a paid Google Gemini API plan. 
            Free tier users will receive enhanced placeholder images. 
            <a 
              href="https://ai.google.dev/pricing" 
              target="_blank" 
              rel="noopener noreferrer"
              className="underline hover:text-yellow-100 ml-1"
            >
              Upgrade your plan
            </a> to enable real AI image generation.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">Image Description</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Describe the image you want to create... (e.g., A futuristic city at sunset)"
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
              <ImageIcon className="w-5 h-5" />
              <span>Generate Image</span>
            </>
          )}
        </button>
      </div>

      {imageUrl && (
        <div className="glass rounded-xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold">Generated Image</h2>
            <div className="flex space-x-2">
              <button
                onClick={handleUploadClick}
                className="flex items-center space-x-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors"
              >
                <Upload className="w-4 h-4" />
                <span>Upload</span>
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center space-x-2 px-4 py-2 gradient-primary rounded-lg hover:shadow-lg hover:shadow-purple-500/50 transition-all"
              >
                <Download className="w-4 h-4" />
                <span>Download</span>
              </button>
            </div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <img src={imageUrl.startsWith('http') ? imageUrl : `http://localhost:8000${imageUrl}`} alt="Generated" className="w-full rounded-lg" />
          </div>
        </div>
      )}

      {/* Hidden file input for upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  )
}
