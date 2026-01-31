import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Plus, Search, Filter, FileText, Image, Music, Video, MoreVertical, Edit, Trash2, X, Save } from 'lucide-react'
import api from '../lib/api'

interface ContentItem {
  id: string
  type: string
  title: string
  content: string
  workflow_state: string
  created_at: string
  is_published: boolean
  content_metadata?: {
    author?: string
    description?: string
  }
}

export default function ContentList() {
  const [content, setContent] = useState<ContentItem[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [showMenu, setShowMenu] = useState<string | null>(null)
  const [editingContent, setEditingContent] = useState<ContentItem | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<ContentItem | null>(null)

  useEffect(() => {
    loadContent()
  }, [filterType])

  const loadContent = async () => {
    try {
      setLoading(true)
      const response = await api.get('/content/', {
        params: {
          limit: 100
        }
      })
      setContent(response.data.items || [])
    } catch (error) {
      console.error('Failed to load content:', error)
      setContent([])
    } finally {
      setLoading(false)
    }
  }

  const handleEdit = (item: ContentItem) => {
    setEditingContent(item)
    setShowMenu(null)
  }

  const handleSaveEdit = async () => {
    if (!editingContent) return

    try {
      await api.put(`/content/${editingContent.id}`, {
        title: editingContent.title,
        content: editingContent.content,
        description: editingContent.content_metadata?.description
      })
      
      setContent(content.map(item => 
        item.id === editingContent.id ? editingContent : item
      ))
      
      setEditingContent(null)
      alert('Content updated successfully!')
    } catch (error: any) {
      console.error('Failed to update content:', error)
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || 'Failed to update content. Please try again.'
      alert(errorMsg)
    }
  }

  const handleDelete = async (item: ContentItem) => {
    try {
      await api.delete(`/content/${item.id}`)
      setContent(content.filter(c => c.id !== item.id))
      setShowDeleteConfirm(null)
      alert('Content deleted successfully!')
    } catch (error: any) {
      console.error('Failed to delete content:', error)
      const errorMsg = error.response?.data?.detail || error.response?.data?.message || 'Failed to delete content. Please try again.'
      alert(errorMsg)
    }
  }

  const filteredContent = content.filter((item) =>
    item.title.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text':
        return <FileText className="w-5 h-5" />
      case 'image':
        return <Image className="w-5 h-5" />
      case 'audio':
        return <Music className="w-5 h-5" />
      case 'video':
        return <Video className="w-5 h-5" />
      default:
        return <FileText className="w-5 h-5" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text':
        return 'from-blue-500 to-cyan-500'
      case 'image':
        return 'from-purple-500 to-pink-500'
      case 'audio':
        return 'from-green-500 to-emerald-500'
      case 'video':
        return 'from-orange-500 to-red-500'
      default:
        return 'from-slate-500 to-slate-600'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">Content</h1>
          <p className="text-slate-400">Manage all your content in one place</p>
        </div>
        <Link
          to="/app/content/create"
          className="flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all"
        >
          <Plus className="w-5 h-5" />
          <span>Create Content</span>
        </Link>
      </div>

      {/* Filters */}
      <div className="glass rounded-xl p-6">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search content..."
              className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
            />
          </div>

          {/* Type Filter */}
          <div className="flex items-center space-x-2">
            <Filter className="w-5 h-5 text-slate-400" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
            >
              <option value="all">All Types</option>
              <option value="text">Text</option>
              <option value="image">Image</option>
              <option value="audio">Audio</option>
              <option value="video">Video</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content Grid */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400 mt-4">Loading content...</p>
        </div>
      ) : filteredContent.length === 0 ? (
        <div className="glass rounded-xl p-12 text-center">
          <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">No content yet</h3>
          <p className="text-slate-400 mb-6">Create your first content item to get started</p>
          <Link
            to="/app/content/create"
            className="inline-flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all"
          >
            <Plus className="w-5 h-5" />
            <span>Create Content</span>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredContent.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="glass rounded-xl p-6 glass-hover group relative"
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getTypeColor(item.type)} flex items-center justify-center`}>
                  {getTypeIcon(item.type)}
                </div>
                <div className="relative">
                  <button 
                    onClick={() => setShowMenu(showMenu === item.id ? null : item.id)}
                    className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                  >
                    <MoreVertical className="w-5 h-5" />
                  </button>
                  
                  {/* Dropdown Menu */}
                  {showMenu === item.id && (
                    <div className="absolute right-0 mt-2 w-48 glass rounded-lg shadow-xl z-10 border border-slate-700">
                      <button
                        onClick={() => handleEdit(item)}
                        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-slate-800 transition-colors text-left rounded-t-lg"
                      >
                        <Edit className="w-4 h-4" />
                        <span>Edit</span>
                      </button>
                      <button
                        onClick={() => {
                          setShowDeleteConfirm(item)
                          setShowMenu(null)
                        }}
                        className="w-full flex items-center gap-2 px-4 py-3 hover:bg-red-500/10 text-red-400 transition-colors text-left rounded-b-lg"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              <h3 className="font-semibold mb-2 line-clamp-2">{item.title}</h3>

              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400 capitalize">{item.type}</span>
                <span className={`px-2 py-1 rounded-full text-xs ${
                  item.is_published
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {item.is_published ? 'Published' : 'Draft'}
                </span>
              </div>

              <div className="mt-4 pt-4 border-t border-slate-800">
                <p className="text-xs text-slate-500">
                  {new Date(item.created_at).toLocaleDateString()}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      )}

      {/* Edit Modal */}
      <AnimatePresence>
        {editingContent && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setEditingContent(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 glass-sidebar border-b border-slate-800 p-6 flex items-center justify-between z-10">
                <h2 className="text-2xl font-bold">Edit Content</h2>
                <button
                  onClick={() => setEditingContent(null)}
                  className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="p-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Title</label>
                  <input
                    type="text"
                    value={editingContent.title}
                    onChange={(e) => setEditingContent({ ...editingContent, title: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Content</label>
                  <textarea
                    value={typeof editingContent.content === 'string' ? editingContent.content : JSON.stringify(editingContent.content)}
                    onChange={(e) => setEditingContent({ ...editingContent, content: e.target.value })}
                    rows={10}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Description</label>
                  <input
                    type="text"
                    value={editingContent.content_metadata?.description || ''}
                    onChange={(e) => setEditingContent({
                      ...editingContent,
                      content_metadata: {
                        ...editingContent.content_metadata,
                        description: e.target.value
                      }
                    })}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>

                <div className="flex gap-3 pt-4">
                  <button
                    onClick={handleSaveEdit}
                    className="flex-1 flex items-center justify-center gap-2 py-3 bg-purple-500 hover:bg-purple-600 rounded-lg font-medium transition-all"
                  >
                    <Save className="w-5 h-5" />
                    <span>Save Changes</span>
                  </button>
                  <button
                    onClick={() => setEditingContent(null)}
                    className="flex-1 py-3 glass hover:bg-slate-800 rounded-lg font-medium transition-all"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Delete Confirmation Modal */}
      <AnimatePresence>
        {showDeleteConfirm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={() => setShowDeleteConfirm(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="glass rounded-2xl max-w-md w-full p-6"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                  <Trash2 className="w-6 h-6 text-red-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold">Delete Content</h3>
                  <p className="text-slate-400 text-sm">This action cannot be undone</p>
                </div>
              </div>

              <p className="text-slate-300 mb-6">
                Are you sure you want to delete "<strong>{showDeleteConfirm.title}</strong>"?
              </p>

              <div className="flex gap-3">
                <button
                  onClick={() => handleDelete(showDeleteConfirm)}
                  className="flex-1 py-3 bg-red-500 hover:bg-red-600 rounded-lg font-medium transition-all"
                >
                  Delete
                </button>
                <button
                  onClick={() => setShowDeleteConfirm(null)}
                  className="flex-1 py-3 glass hover:bg-slate-800 rounded-lg font-medium transition-all"
                >
                  Cancel
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
