import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Search, Filter, TrendingUp, Clock, Heart, MessageCircle, Share2, 
  FileText, Image, Music, Video, Eye, Bookmark, MoreVertical,
  Sparkles, Globe, Hash, Users, X, ExternalLink
} from 'lucide-react'
import api from '../lib/api'
import { getMediaUrl, isImageUrl, isVideoUrl, isAudioUrl } from '../lib/mediaUtils'

interface ContentItem {
  id: string
  type: string
  title: string
  content: string
  content_metadata: {
    author: string
    description?: string
    word_count?: number
  }
  user_id: string
  tags: string[]
  created_at: string
  is_published: boolean
  engagement?: {
    views: number
    likes: number
    comments: number
    shares: number
  }
}

interface TrendingTopic {
  tag: string
  count: number
  growth: string
}

export default function ContentDiscovery() {
  const [content, setContent] = useState<ContentItem[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<string>('all')
  const [sortBy, setSortBy] = useState<string>('recent')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')
  const [selectedContent, setSelectedContent] = useState<ContentItem | null>(null)

  const trendingTopics: TrendingTopic[] = [
    { tag: 'AI', count: 1234, growth: '+45%' },
    { tag: 'Technology', count: 892, growth: '+32%' },
    { tag: 'Marketing', count: 756, growth: '+28%' },
    { tag: 'Design', count: 645, growth: '+21%' },
    { tag: 'Business', count: 534, growth: '+18%' },
  ]

  useEffect(() => {
    loadDiscoveryContent()
  }, [filterType, sortBy])

  const loadDiscoveryContent = async () => {
    try {
      setLoading(true)
      // Load all published content from all users
      const response = await api.get('/content/', {
        params: {
          all_users: true,
          limit: 50
        }
      })
      
      console.log('Content loaded:', response.data.items)
      
      // Use mock engagement data for now (engagement endpoints not implemented yet)
      const contentWithEngagement = (response.data.items || []).map((item: ContentItem) => ({
        ...item,
        engagement: {
          views: Math.floor(Math.random() * 5000) + 100,
          likes: Math.floor(Math.random() * 500) + 10,
          comments: Math.floor(Math.random() * 100) + 5,
          shares: Math.floor(Math.random() * 200) + 5,
        }
      }))
      
      setContent(contentWithEngagement)
    } catch (error) {
      console.error('Failed to load discovery content:', error)
      setContent([])
    } finally {
      setLoading(false)
    }
  }

  const trackEngagement = async (contentId: string, action: 'view' | 'like' | 'share') => {
    // Engagement tracking disabled for now (endpoints not implemented)
    // Just update local state with mock data
    setContent(prevContent => 
      prevContent.map(item => {
        if (item.id === contentId && item.engagement) {
          return {
            ...item,
            engagement: {
              ...item.engagement,
              [action === 'view' ? 'views' : action === 'like' ? 'likes' : 'shares']: 
                item.engagement[action === 'view' ? 'views' : action === 'like' ? 'likes' : 'shares'] + 1
            }
          }
        }
        return item
      })
    )
  }

  const handleContentClick = (item: ContentItem) => {
    setSelectedContent(item)
    trackEngagement(item.id, 'view')
  }

  const handleLike = (contentId: string) => {
    trackEngagement(contentId, 'like')
  }

  const handleShare = (contentId: string) => {
    trackEngagement(contentId, 'share')
  }

  const filteredContent = content
    .filter((item) => {
      const matchesSearch = item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           item.content_metadata.description?.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesType = filterType === 'all' || item.type === filterType
      return matchesSearch && matchesType
    })
    .sort((a, b) => {
      if (sortBy === 'recent') {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      } else if (sortBy === 'popular') {
        return (b.engagement?.views || 0) - (a.engagement?.views || 0)
      } else if (sortBy === 'trending') {
        return (b.engagement?.likes || 0) - (a.engagement?.likes || 0)
      }
      return 0
    })

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <Globe className="w-10 h-10 text-purple-400" />
            Discover Content
          </h1>
          <p className="text-slate-400">Explore trending content from creators worldwide</p>
        </div>
      </div>

      {/* Trending Topics Bar */}
      <div className="glass rounded-xl p-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold">Trending Topics</h3>
        </div>
        <div className="flex flex-wrap gap-2">
          {trendingTopics.map((topic) => (
            <button
              key={topic.tag}
              className="px-4 py-2 bg-slate-900/50 hover:bg-slate-800 rounded-full border border-slate-700 hover:border-purple-500 transition-all group"
            >
              <div className="flex items-center gap-2">
                <Hash className="w-4 h-4 text-purple-400" />
                <span className="font-medium">{topic.tag}</span>
                <span className="text-xs text-slate-400">{formatNumber(topic.count)}</span>
                <span className="text-xs text-green-400">{topic.growth}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Filters and Search */}
      <div className="glass rounded-xl p-6">
        <div className="flex flex-col lg:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search content, topics, creators..."
              className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
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

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
            >
              <option value="recent">Most Recent</option>
              <option value="popular">Most Popular</option>
              <option value="trending">Trending</option>
            </select>

            <div className="flex gap-2">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-3 rounded-lg transition-all ${
                  viewMode === 'grid'
                    ? 'bg-purple-500 text-white'
                    : 'bg-slate-900/50 hover:bg-slate-800'
                }`}
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                </svg>
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-3 rounded-lg transition-all ${
                  viewMode === 'list'
                    ? 'bg-purple-500 text-white'
                    : 'bg-slate-900/50 hover:bg-slate-800'
                }`}
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Content Grid/List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400 mt-4">Loading content...</p>
        </div>
      ) : filteredContent.length === 0 ? (
        <div className="glass rounded-xl p-12 text-center">
          <Sparkles className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-semibold mb-2">No content found</h3>
          <p className="text-slate-400">Try adjusting your filters or search query</p>
        </div>
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredContent.map((item, index) => (
            <ContentCard 
              key={item.id} 
              item={item} 
              index={index} 
              onClick={() => handleContentClick(item)}
              onLike={() => handleLike(item.id)}
              onShare={() => handleShare(item.id)}
            />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {filteredContent.map((item, index) => (
            <ContentListItem 
              key={item.id} 
              item={item} 
              index={index} 
              onClick={() => handleContentClick(item)}
              onLike={() => handleLike(item.id)}
              onShare={() => handleShare(item.id)}
            />
          ))}
        </div>
      )}

      {/* Content Detail Modal */}
      {selectedContent && (
        <ContentDetailModal 
          content={selectedContent} 
          onClose={() => setSelectedContent(null)}
          onLike={() => handleLike(selectedContent.id)}
          onShare={() => handleShare(selectedContent.id)}
        />
      )}
    </div>
  )
}

function ContentCard({ item, index, onClick, onLike, onShare }: { 
  item: ContentItem; 
  index: number; 
  onClick: () => void;
  onLike: () => void;
  onShare: () => void;
}) {
  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return <FileText className="w-5 h-5" />
      case 'image': return <Image className="w-5 h-5" />
      case 'audio': return <Music className="w-5 h-5" />
      case 'video': return <Video className="w-5 h-5" />
      default: return <FileText className="w-5 h-5" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return 'from-blue-500 to-cyan-500'
      case 'image': return 'from-purple-500 to-pink-500'
      case 'audio': return 'from-green-500 to-emerald-500'
      case 'video': return 'from-orange-500 to-red-500'
      default: return 'from-slate-500 to-slate-600'
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="glass rounded-xl overflow-hidden glass-hover group cursor-pointer"
      onClick={onClick}
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <p className="font-medium">{item.content_metadata.author}</p>
              <p className="text-xs text-slate-400">@{item.user_id}</p>
            </div>
          </div>
          <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
            <MoreVertical className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {/* Image Preview for image type */}
        {item.type === 'image' && typeof item.content === 'string' && (
          <div className="mb-4 rounded-lg overflow-hidden bg-slate-900/50 border border-slate-800">
            <img
              src={getMediaUrl(item.content)}
              alt={item.title}
              className="w-full h-48 object-cover"
              onError={(e) => {
                e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23334155" width="400" height="300"/%3E%3Ctext fill="%23cbd5e1" font-family="sans-serif" font-size="18" x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle"%3EImage Preview%3C/text%3E%3C/svg%3E'
              }}
            />
          </div>
        )}

        <div className="flex items-center gap-2 mb-3">
          <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${getTypeColor(item.type)} flex items-center justify-center`}>
            {getTypeIcon(item.type)}
          </div>
          <span className="text-xs text-slate-400 capitalize">{item.type}</span>
        </div>

        <h3 className="font-semibold mb-2 line-clamp-2 group-hover:text-purple-400 transition-colors">
          {item.title}
        </h3>
        
        {item.content_metadata.description && (
          <p className="text-sm text-slate-400 line-clamp-2 mb-3">
            {item.content_metadata.description}
          </p>
        )}

        {/* Tags */}
        {item.tags && item.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-3">
            {item.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="px-2 py-1 bg-slate-900/50 rounded-full text-xs text-purple-400"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Engagement Stats */}
        <div className="flex items-center justify-between text-sm text-slate-400 pt-3 border-t border-slate-800">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1">
              <Eye className="w-4 h-4" />
              <span>{formatNumber(item.engagement?.views || 0)}</span>
            </div>
            <div className="flex items-center gap-1">
              <Heart className="w-4 h-4" />
              <span>{formatNumber(item.engagement?.likes || 0)}</span>
            </div>
            <div className="flex items-center gap-1">
              <MessageCircle className="w-4 h-4" />
              <span>{formatNumber(item.engagement?.comments || 0)}</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={(e) => {
                e.stopPropagation()
                onLike()
              }}
              className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors hover:text-red-400"
            >
              <Heart className="w-4 h-4" />
            </button>
            <button className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors">
              <Bookmark className="w-4 h-4" />
            </button>
            <button 
              onClick={(e) => {
                e.stopPropagation()
                onShare()
              }}
              className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors hover:text-blue-400"
            >
              <Share2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-slate-900/30 border-t border-slate-800">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>{new Date(item.created_at).toLocaleDateString()}</span>
          </div>
          {item.content_metadata.word_count && (
            <span>{item.content_metadata.word_count} words</span>
          )}
        </div>
      </div>
    </motion.div>
  )
}

function ContentListItem({ item, index, onClick, onLike, onShare }: { 
  item: ContentItem; 
  index: number; 
  onClick: () => void;
  onLike: () => void;
  onShare: () => void;
}) {
  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return <FileText className="w-5 h-5" />
      case 'image': return <Image className="w-5 h-5" />
      case 'audio': return <Music className="w-5 h-5" />
      case 'video': return <Video className="w-5 h-5" />
      default: return <FileText className="w-5 h-5" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return 'from-blue-500 to-cyan-500'
      case 'image': return 'from-purple-500 to-pink-500'
      case 'audio': return 'from-green-500 to-emerald-500'
      case 'video': return 'from-orange-500 to-red-500'
      default: return 'from-slate-500 to-slate-600'
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.03 }}
      className="glass rounded-xl p-6 glass-hover group cursor-pointer"
      onClick={onClick}
    >
      <div className="flex items-start gap-4">
        {/* Media Preview for images */}
        {item.type === 'image' && typeof item.content === 'string' && (
          <div className="w-32 h-32 rounded-lg overflow-hidden bg-slate-900/50 border border-slate-800 flex-shrink-0">
            <img
              src={getMediaUrl(item.content)}
              alt={item.title}
              className="w-full h-full object-cover"
              onError={(e) => {
                e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="200" height="200"%3E%3Crect fill="%23334155" width="200" height="200"/%3E%3Ctext fill="%23cbd5e1" font-family="sans-serif" font-size="16" x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle"%3EImage%3C/text%3E%3C/svg%3E'
              }}
            />
          </div>
        )}

        {/* Type Icon for non-images */}
        {item.type !== 'image' && (
          <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getTypeColor(item.type)} flex items-center justify-center flex-shrink-0`}>
            {getTypeIcon(item.type)}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between mb-2">
            <div className="flex-1">
              <h3 className="font-semibold mb-1 group-hover:text-purple-400 transition-colors">
                {item.title}
              </h3>
              <div className="flex items-center gap-3 text-sm text-slate-400">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <Users className="w-3 h-3" />
                  </div>
                  <span>{item.content_metadata.author}</span>
                </div>
                <span>•</span>
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  <span>{new Date(item.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
            <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
              <MoreVertical className="w-5 h-5" />
            </button>
          </div>

          {item.content_metadata.description && (
            <p className="text-sm text-slate-400 mb-3 line-clamp-2">
              {item.content_metadata.description}
            </p>
          )}

          {/* Tags */}
          {item.tags && item.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-3">
              {item.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-slate-900/50 rounded-full text-xs text-purple-400"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Engagement */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6 text-sm text-slate-400">
              <div className="flex items-center gap-1">
                <Eye className="w-4 h-4" />
                <span>{formatNumber(item.engagement?.views || 0)}</span>
              </div>
              <div className="flex items-center gap-1">
                <Heart className="w-4 h-4" />
                <span>{formatNumber(item.engagement?.likes || 0)}</span>
              </div>
              <div className="flex items-center gap-1">
                <MessageCircle className="w-4 h-4" />
                <span>{formatNumber(item.engagement?.comments || 0)}</span>
              </div>
              <div className="flex items-center gap-1">
                <Share2 className="w-4 h-4" />
                <span>{formatNumber(item.engagement?.shares || 0)}</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button 
                onClick={(e) => {
                  e.stopPropagation()
                  onLike()
                }}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors hover:text-red-400"
              >
                <Heart className="w-4 h-4" />
              </button>
              <button className="p-2 hover:bg-slate-800 rounded-lg transition-colors">
                <Bookmark className="w-4 h-4" />
              </button>
              <button 
                onClick={(e) => {
                  e.stopPropagation()
                  onShare()
                }}
                className="p-2 hover:bg-slate-800 rounded-lg transition-colors hover:text-blue-400"
              >
                <Share2 className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}


function ContentDetailModal({ content, onClose, onLike, onShare }: { 
  content: ContentItem; 
  onClose: () => void;
  onLike: () => void;
  onShare: () => void;
}) {
  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return <FileText className="w-6 h-6" />
      case 'image': return <Image className="w-6 h-6" />
      case 'audio': return <Music className="w-6 h-6" />
      case 'video': return <Video className="w-6 h-6" />
      default: return <FileText className="w-6 h-6" />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'text': return 'from-blue-500 to-cyan-500'
      case 'image': return 'from-purple-500 to-pink-500'
      case 'audio': return 'from-green-500 to-emerald-500'
      case 'video': return 'from-orange-500 to-red-500'
      default: return 'from-slate-500 to-slate-600'
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="glass rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="sticky top-0 glass-sidebar border-b border-slate-800 p-6 flex items-center justify-between z-10">
            <div className="flex items-center gap-4">
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${getTypeColor(content.type)} flex items-center justify-center`}>
                {getTypeIcon(content.type)}
              </div>
              <div>
                <h2 className="text-2xl font-bold">{content.title}</h2>
                <div className="flex items-center gap-3 text-sm text-slate-400 mt-1">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      <Users className="w-3 h-3" />
                    </div>
                    <span>{content.content_metadata.author}</span>
                  </div>
                  <span>•</span>
                  <div className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    <span>{new Date(content.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Media Preview */}
            {content.type === 'image' && typeof content.content === 'string' && isImageUrl(content.content) && (
              <div className="rounded-xl overflow-hidden bg-slate-900/50 border border-slate-800">
                <img
                  src={getMediaUrl(content.content)}
                  alt={content.title}
                  className="w-full h-auto max-h-[500px] object-contain"
                  onError={(e) => {
                    e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="800" height="600"%3E%3Crect fill="%23334155" width="800" height="600"/%3E%3Ctext fill="%23cbd5e1" font-family="sans-serif" font-size="24" x="50%25" y="50%25" text-anchor="middle" dominant-baseline="middle"%3EImage Not Available%3C/text%3E%3C/svg%3E'
                  }}
                />
              </div>
            )}

            {content.type === 'video' && typeof content.content === 'string' && isVideoUrl(content.content) && (
              <div className="rounded-xl overflow-hidden bg-slate-900/50 border border-slate-800">
                <video
                  src={getMediaUrl(content.content)}
                  controls
                  className="w-full h-auto max-h-[500px]"
                >
                  Your browser does not support the video tag.
                </video>
              </div>
            )}

            {content.type === 'audio' && typeof content.content === 'string' && isAudioUrl(content.content) && (
              <div className="rounded-xl bg-slate-900/50 border border-slate-800 p-6">
                <audio
                  src={getMediaUrl(content.content)}
                  controls
                  className="w-full"
                >
                  Your browser does not support the audio tag.
                </audio>
              </div>
            )}

            {/* Text Content */}
            {content.type === 'text' && (
              <div className="prose prose-invert max-w-none">
                <div className="bg-slate-900/50 rounded-xl p-6 border border-slate-800">
                  <p className="whitespace-pre-wrap text-slate-300 leading-relaxed">
                    {typeof content.content === 'string' ? content.content : JSON.stringify(content.content)}
                  </p>
                </div>
              </div>
            )}

            {/* Description */}
            {content.content_metadata.description && (
              <div>
                <h3 className="text-lg font-semibold mb-2">Description</h3>
                <p className="text-slate-400">{content.content_metadata.description}</p>
              </div>
            )}

            {/* Tags */}
            {content.tags && content.tags.length > 0 && (
              <div>
                <h3 className="text-lg font-semibold mb-3">Tags</h3>
                <div className="flex flex-wrap gap-2">
                  {content.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-3 py-1.5 bg-slate-900/50 rounded-full text-sm text-purple-400 border border-slate-800"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Engagement Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <Eye className="w-4 h-4" />
                  <span className="text-sm">Views</span>
                </div>
                <p className="text-2xl font-bold">{formatNumber(content.engagement?.views || 0)}</p>
              </div>
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <Heart className="w-4 h-4" />
                  <span className="text-sm">Likes</span>
                </div>
                <p className="text-2xl font-bold">{formatNumber(content.engagement?.likes || 0)}</p>
              </div>
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <MessageCircle className="w-4 h-4" />
                  <span className="text-sm">Comments</span>
                </div>
                <p className="text-2xl font-bold">{formatNumber(content.engagement?.comments || 0)}</p>
              </div>
              <div className="bg-slate-900/50 rounded-xl p-4 border border-slate-800">
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <Share2 className="w-4 h-4" />
                  <span className="text-sm">Shares</span>
                </div>
                <p className="text-2xl font-bold">{formatNumber(content.engagement?.shares || 0)}</p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pt-4 border-t border-slate-800">
              <button 
                onClick={onLike}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-purple-500 hover:bg-purple-600 rounded-lg font-medium transition-all"
              >
                <Heart className="w-5 h-5" />
                <span>Like</span>
              </button>
              <button className="flex-1 flex items-center justify-center gap-2 py-3 glass hover:bg-slate-800 rounded-lg font-medium transition-all">
                <MessageCircle className="w-5 h-5" />
                <span>Comment</span>
              </button>
              <button 
                onClick={onShare}
                className="flex-1 flex items-center justify-center gap-2 py-3 glass hover:bg-slate-800 rounded-lg font-medium transition-all"
              >
                <Share2 className="w-5 h-5" />
                <span>Share</span>
              </button>
              {typeof content.content === 'string' && (content.type === 'image' || content.type === 'video' || content.type === 'audio') && (
                <a
                  href={content.content}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-3 glass hover:bg-slate-800 rounded-lg transition-all"
                >
                  <ExternalLink className="w-5 h-5" />
                </a>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
