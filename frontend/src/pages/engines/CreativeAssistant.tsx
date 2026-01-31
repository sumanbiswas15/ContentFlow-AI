import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Sparkles, Loader2, Upload, X, FileText, Image as ImageIcon, File } from 'lucide-react'
import api from '../../lib/api'

export default function CreativeAssistant() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [fileContents, setFileContents] = useState<string[]>([])

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return

    const newFiles: File[] = []
    const newContents: string[] = []

    for (const file of files) {
      // Check file type
      const isImage = file.type.startsWith('image/')
      const isText = file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md')
      const isPDF = file.type === 'application/pdf' || file.name.endsWith('.pdf')
      const isDoc = file.type === 'application/msword' || 
                    file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
                    file.name.endsWith('.doc') || file.name.endsWith('.docx')

      if (!isImage && !isText && !isPDF && !isDoc) {
        alert(`File ${file.name} is not supported. Please upload images, text files, PDF, or DOC files.`)
        continue
      }

      newFiles.push(file)

      // Read file content for text files
      if (isText) {
        const content = await readFileAsText(file)
        newContents.push(content)
      } else if (isPDF || isDoc) {
        // For PDF and DOC files, we'll note that they're uploaded
        // In a production app, you'd send these to the backend for processing
        newContents.push(`[${file.name} - Document content will be processed]`)
      } else {
        newContents.push('') // Empty for images
      }
    }

    setUploadedFiles([...uploadedFiles, ...newFiles])
    setFileContents([...fileContents, ...newContents])
  }

  const readFileAsText = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target?.result as string)
      reader.onerror = reject
      reader.readAsText(file)
    })
  }

  const removeFile = (index: number) => {
    setUploadedFiles(uploadedFiles.filter((_, i) => i !== index))
    setFileContents(fileContents.filter((_, i) => i !== index))
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    setLoading(true)
    try {
      // Build context with uploaded file contents
      let contextPrompt = prompt
      if (fileContents.length > 0) {
        const textContents = fileContents.filter(c => c).join('\n\n---\n\n')
        if (textContents) {
          contextPrompt = `${prompt}\n\nContext from uploaded files:\n${textContents}`
        }
      }

      // If no session, create one first
      if (!sessionId) {
        const sessionResponse = await api.post('/engines/creative/start-session', {
          session_type: 'ideation',
          topic: contextPrompt,
          target_audience: 'general',
          brand_voice: 'professional',
          goals: ['generate ideas']
        })
        setSessionId(sessionResponse.data.session_id)
      }

      // Get suggestions using the session
      const currentSessionId = sessionId || (await api.post('/engines/creative/start-session', {
        session_type: 'ideation',
        topic: contextPrompt,
        target_audience: 'general',
        brand_voice: 'professional',
        goals: ['generate ideas']
      })).data.session_id

      const response = await api.post(`/engines/creative/${currentSessionId}/suggestions`, {
        suggestion_type: 'idea',
        context: contextPrompt,
        count: 5
      })
      
      // Format suggestions into readable text
      const suggestions = response.data.suggestions || []
      const formattedResult = suggestions.map((s: { content: string; rationale: string; confidence_score: number }, i: number) => 
        `${i + 1}. ${s.content}\n   Rationale: ${s.rationale}\n   Confidence: ${(s.confidence_score * 100).toFixed(0)}%`
      ).join('\n\n')
      
      setResult(formattedResult || 'No suggestions available.')
    } catch (error: unknown) {
      console.error('Generation failed:', error)
      const err = error as { response?: { data?: { detail?: string } } }
      const errorMsg = err.response?.data?.detail || 'Failed to get suggestions. Please try again.'
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
          <h1 className="text-4xl font-bold">Creative Assistant</h1>
          <p className="text-slate-400">Get creative suggestions and ideas</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">What do you need help with?</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all min-h-[120px]"
            placeholder="Ask for creative ideas... (e.g., Give me 5 video ideas for a tech startup)"
          />
        </div>

        {/* File Upload Section */}
        <div>
          <label className="block text-sm font-medium mb-2">Upload Reference Files (Optional)</label>
          <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 text-center hover:border-purple-500 transition-colors">
            <input
              type="file"
              id="file-upload"
              multiple
              accept="image/*,.txt,.md,.pdf,.doc,.docx,text/*,application/pdf,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              onChange={handleFileUpload}
              className="hidden"
            />
            <label
              htmlFor="file-upload"
              className="cursor-pointer flex flex-col items-center space-y-2"
            >
              <Upload className="w-8 h-8 text-slate-400" />
              <span className="text-sm text-slate-400">
                Click to upload files
              </span>
              <span className="text-xs text-slate-500">
                Supports: Images, Text (TXT, MD), PDF, Word (DOC, DOCX)
              </span>
            </label>
          </div>

          {/* Uploaded Files List */}
          {uploadedFiles.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="text-sm font-medium">Uploaded Files:</p>
              {uploadedFiles.map((file, index) => {
                const isImage = file.type.startsWith('image/')
                const isPDF = file.type === 'application/pdf' || file.name.endsWith('.pdf')
                const isDoc = file.type === 'application/msword' || 
                              file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
                              file.name.endsWith('.doc') || file.name.endsWith('.docx')
                
                return (
                  <div
                    key={index}
                    className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3 border border-slate-700"
                  >
                    <div className="flex items-center space-x-3">
                      {isImage ? (
                        <ImageIcon className="w-5 h-5 text-purple-400" />
                      ) : isPDF ? (
                        <File className="w-5 h-5 text-red-400" />
                      ) : isDoc ? (
                        <File className="w-5 h-5 text-blue-400" />
                      ) : (
                        <FileText className="w-5 h-5 text-green-400" />
                      )}
                      <div>
                        <p className="text-sm font-medium">{file.name}</p>
                        <p className="text-xs text-slate-400">
                          {(file.size / 1024).toFixed(2)} KB
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="p-1 hover:bg-slate-800 rounded transition-colors"
                    >
                      <X className="w-4 h-4 text-slate-400 hover:text-red-400" />
                    </button>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading || !prompt.trim()}
          className="w-full flex items-center justify-center space-x-2 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Getting Suggestions...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Get Suggestions</span>
            </>
          )}
        </button>
      </div>

      {result && (
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold">Creative Suggestions</h2>
          <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700">
            <p className="whitespace-pre-wrap">{result}</p>
          </div>
        </div>
      )}
    </div>
  )
}
