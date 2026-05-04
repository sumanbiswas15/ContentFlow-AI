import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Save, ArrowLeft, FileText, Image, Music, Video, Upload } from 'lucide-react'
import api from '../lib/api'

export default function ContentCreate() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState({
    type: 'text',
    title: '',
    content: '',
    author: '',
    description: '',
    language: 'en',
    tags: '',
    // Image-specific fields
    imageUrl: '',
    imageFile: null as File | null,
    // Audio-specific fields
    audioUrl: '',
    audioFile: null as File | null,
    duration: '',
    // Video-specific fields
    videoUrl: '',
    videoFile: null as File | null,
    videoDuration: '',
    resolution: '',
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Build payload with only the fields the backend expects
      const payload: {
        type: string
        title: string
        content: string
        author: string
        description: string
        language: string
        tags: string[]
        workflow_state: string
      } = {
        type: formData.type,
        title: formData.title,
        content: formData.content,
        author: formData.author,
        description: formData.description || '',
        language: formData.language,
        tags: formData.tags.split(',').map((tag) => tag.trim()).filter(Boolean),
        workflow_state: 'create'
      }

      // Ensure content field is populated based on type
      if (formData.type === 'image' && !payload.content) {
        payload.content = formData.imageUrl || formData.imageFile?.name || ''
      } else if (formData.type === 'audio' && !payload.content) {
        payload.content = formData.audioUrl || formData.audioFile?.name || ''
      } else if (formData.type === 'video' && !payload.content) {
        payload.content = formData.videoUrl || formData.videoFile?.name || ''
      }

      await api.post('/content/', payload)
      navigate('/app/content')
    } catch (error) {
      const err = error as { response?: { data?: { detail?: string } } }
      console.error('Failed to create content:', error)
      alert(err.response?.data?.detail || 'Failed to create content')
    } finally {
      setLoading(false)
    }
  }

  const contentTypes = [
    { value: 'text', label: 'Text', icon: FileText, color: 'from-blue-500 to-cyan-500' },
    { value: 'image', label: 'Image', icon: Image, color: 'from-purple-500 to-pink-500' },
    { value: 'audio', label: 'Audio', icon: Music, color: 'from-green-500 to-emerald-500' },
    { value: 'video', label: 'Video', icon: Video, color: 'from-orange-500 to-red-500' },
  ]

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <button
          onClick={() => navigate('/app/content')}
          className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-6 h-6" />
        </button>
        <div>
          <h1 className="text-4xl font-bold">Create Content</h1>
          <p className="text-slate-400">Start with a new content item</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Content Type Selection */}
        <div className="glass rounded-xl p-6">
          <label className="block text-sm font-medium mb-4">Content Type</label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {contentTypes.map((type) => (
              <button
                key={type.value}
                type="button"
                onClick={() => setFormData({ ...formData, type: type.value })}
                className={`p-4 rounded-lg border-2 transition-all ${
                  formData.type === type.value
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-700 hover:border-slate-600'
                }`}
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${type.color} flex items-center justify-center mx-auto mb-2`}>
                  <type.icon className="w-6 h-6" />
                </div>
                <p className="text-sm font-medium">{type.label}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Basic Information */}
        <div className="glass rounded-xl p-6 space-y-4">
          <h2 className="text-xl font-semibold mb-4">Basic Information</h2>

          <div>
            <label className="block text-sm font-medium mb-2">Title *</label>
            <input
              type="text"
              value={formData.title}
              onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
              placeholder="Enter content title"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Author *</label>
            <input
              type="text"
              value={formData.author}
              onChange={(e) => setFormData({ ...formData, author: e.target.value })}
              className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
              placeholder="Author name"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all resize-none"
              rows={3}
              placeholder="Brief description of the content"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Language</label>
              <select
                value={formData.language}
                onChange={(e) => setFormData({ ...formData, language: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
              >
                <option value="en">English</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
                <option value="it">Italian</option>
                <option value="pt">Portuguese</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Tags</label>
              <input
                type="text"
                value={formData.tags}
                onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                placeholder="tag1, tag2, tag3"
              />
            </div>
          </div>
        </div>

        {/* Content - Dynamic based on type */}
        {formData.type === 'text' && (
          <div className="glass rounded-xl p-6">
            <label className="block text-sm font-medium mb-2">Text Content *</label>
            <textarea
              value={formData.content}
              onChange={(e) => setFormData({ ...formData, content: e.target.value })}
              className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all resize-none"
              rows={12}
              placeholder="Enter your text content here..."
              required
            />
            <p className="text-xs text-slate-500 mt-2">
              {formData.content.length} characters
            </p>
          </div>
        )}

        {formData.type === 'image' && (
          <div className="glass rounded-xl p-6 space-y-4">
            <h2 className="text-xl font-semibold mb-4">Image Details</h2>
            
            <div>
              <label className="block text-sm font-medium mb-2">Image URL</label>
              <input
                type="url"
                value={formData.imageUrl}
                onChange={(e) => setFormData({ ...formData, imageUrl: e.target.value, content: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                placeholder="https://example.com/image.jpg or use file upload below"
              />
            </div>

            <div className="text-center text-slate-400">OR</div>

            <div>
              <label className="block text-sm font-medium mb-2">Upload Image File</label>
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-purple-500 transition-all">
                <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) {
                      setFormData({ ...formData, imageFile: file, content: file.name })
                    }
                  }}
                  className="hidden"
                  id="image-upload"
                />
                <label htmlFor="image-upload" className="cursor-pointer">
                  <span className="text-purple-400 hover:text-purple-300">Click to upload</span>
                  <span className="text-slate-400"> or drag and drop</span>
                </label>
                <p className="text-xs text-slate-500 mt-2">PNG, JPG, GIF up to 10MB</p>
                {formData.imageFile && (
                  <p className="text-sm text-green-400 mt-2">Selected: {formData.imageFile.name}</p>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Alt Text / Caption</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all resize-none"
                rows={3}
                placeholder="Describe the image for accessibility and SEO"
              />
            </div>
          </div>
        )}

        {formData.type === 'audio' && (
          <div className="glass rounded-xl p-6 space-y-4">
            <h2 className="text-xl font-semibold mb-4">Audio Details</h2>
            
            <div>
              <label className="block text-sm font-medium mb-2">Audio URL</label>
              <input
                type="url"
                value={formData.audioUrl}
                onChange={(e) => setFormData({ ...formData, audioUrl: e.target.value, content: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                placeholder="https://example.com/audio.mp3 or use file upload below"
              />
            </div>

            <div className="text-center text-slate-400">OR</div>

            <div>
              <label className="block text-sm font-medium mb-2">Upload Audio File</label>
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-purple-500 transition-all">
                <Music className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                <input
                  type="file"
                  accept="audio/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) {
                      setFormData({ ...formData, audioFile: file, content: file.name })
                    }
                  }}
                  className="hidden"
                  id="audio-upload"
                />
                <label htmlFor="audio-upload" className="cursor-pointer">
                  <span className="text-purple-400 hover:text-purple-300">Click to upload</span>
                  <span className="text-slate-400"> or drag and drop</span>
                </label>
                <p className="text-xs text-slate-500 mt-2">MP3, WAV, OGG up to 50MB</p>
                {formData.audioFile && (
                  <p className="text-sm text-green-400 mt-2">Selected: {formData.audioFile.name}</p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Duration (optional)</label>
                <input
                  type="text"
                  value={formData.duration}
                  onChange={(e) => setFormData({ ...formData, duration: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                  placeholder="e.g., 3:45"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Format</label>
                <select
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                >
                  <option value="mp3">MP3</option>
                  <option value="wav">WAV</option>
                  <option value="ogg">OGG</option>
                  <option value="m4a">M4A</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Transcript / Notes</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all resize-none"
                rows={4}
                placeholder="Audio transcript or description"
              />
            </div>
          </div>
        )}

        {formData.type === 'video' && (
          <div className="glass rounded-xl p-6 space-y-4">
            <h2 className="text-xl font-semibold mb-4">Video Details</h2>
            
            <div>
              <label className="block text-sm font-medium mb-2">Video URL</label>
              <input
                type="url"
                value={formData.videoUrl}
                onChange={(e) => setFormData({ ...formData, videoUrl: e.target.value, content: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                placeholder="https://example.com/video.mp4 or use file upload below"
              />
            </div>

            <div className="text-center text-slate-400">OR</div>

            <div>
              <label className="block text-sm font-medium mb-2">Upload Video File</label>
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-8 text-center hover:border-purple-500 transition-all">
                <Video className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) {
                      setFormData({ ...formData, videoFile: file, content: file.name })
                    }
                  }}
                  className="hidden"
                  id="video-upload"
                />
                <label htmlFor="video-upload" className="cursor-pointer">
                  <span className="text-purple-400 hover:text-purple-300">Click to upload</span>
                  <span className="text-slate-400"> or drag and drop</span>
                </label>
                <p className="text-xs text-slate-500 mt-2">MP4, MOV, AVI up to 500MB</p>
                {formData.videoFile && (
                  <p className="text-sm text-green-400 mt-2">Selected: {formData.videoFile.name}</p>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Duration (optional)</label>
                <input
                  type="text"
                  value={formData.videoDuration}
                  onChange={(e) => setFormData({ ...formData, videoDuration: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                  placeholder="e.g., 5:30"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Resolution</label>
                <select
                  value={formData.resolution}
                  onChange={(e) => setFormData({ ...formData, resolution: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                >
                  <option value="">Select resolution</option>
                  <option value="720p">720p (HD)</option>
                  <option value="1080p">1080p (Full HD)</option>
                  <option value="1440p">1440p (2K)</option>
                  <option value="2160p">2160p (4K)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Video Description / Script</label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all resize-none"
                rows={4}
                placeholder="Video description, script, or key points"
              />
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-end space-x-4">
          <button
            type="button"
            onClick={() => navigate('/app/content')}
            className="px-6 py-3 glass rounded-lg font-medium glass-hover"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading}
            className="flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Save className="w-5 h-5" />
            <span>{loading ? 'Creating...' : 'Create Content'}</span>
          </button>
        </div>
      </form>
    </div>
  )
}
