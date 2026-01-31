import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  TrendingUp, Eye, Heart, Share2, MessageCircle,
  BarChart3, PieChart, Activity
} from 'lucide-react'
import {
  LineChart, Line, BarChart, Bar, PieChart as RechartsPie, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import api from '../lib/api'

interface EngagementStats {
  views: number
  likes: number
  shares: number
  comments: number
}

interface ContentTypeStats {
  type: string
  count: number
  engagement: number
}

interface TimeSeriesData {
  date: string
  views: number
  likes: number
  shares: number
}

export default function Analytics() {
  const [loading, setLoading] = useState(true)
  const [totalStats, setTotalStats] = useState<EngagementStats>({
    views: 0,
    likes: 0,
    shares: 0,
    comments: 0
  })
  const [contentTypeStats, setContentTypeStats] = useState<ContentTypeStats[]>([])
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([])

  useEffect(() => {
    loadAnalytics()
  }, [])

  const loadAnalytics = async () => {
    try {
      setLoading(true)
      
      // Load all content to calculate analytics
      const response = await api.get('/content/', {
        params: {
          all_users: true,
          limit: 100
        }
      })

      const content = response.data.items || []
      
      // Calculate total stats
      let totalViews = 0
      let totalLikes = 0
      let totalShares = 0
      let totalComments = 0
      
      const typeMap = new Map<string, { count: number; engagement: number }>()
      const dateMap = new Map<string, { views: number; likes: number; shares: number }>()

      // Process each content item
      // Use mock engagement data for now (engagement endpoints not implemented yet)
      for (const item of content) {
        const mockEngagement = {
          views: Math.floor(Math.random() * 1000) + 100,
          likes: Math.floor(Math.random() * 100) + 10,
          shares: Math.floor(Math.random() * 50) + 5,
          comments: Math.floor(Math.random() * 20) + 2
        }
        
        totalViews += mockEngagement.views
        totalLikes += mockEngagement.likes
        totalShares += mockEngagement.shares
        totalComments += mockEngagement.comments

        const typeData = typeMap.get(item.type) || { count: 0, engagement: 0 }
        typeMap.set(item.type, {
          count: typeData.count + 1,
          engagement: typeData.engagement + mockEngagement.views + mockEngagement.likes
        })

        const date = new Date(item.created_at).toLocaleDateString()
        const dateData = dateMap.get(date) || { views: 0, likes: 0, shares: 0 }
        dateMap.set(date, {
          views: dateData.views + mockEngagement.views,
          likes: dateData.likes + mockEngagement.likes,
          shares: dateData.shares + mockEngagement.shares
        })
      }

      setTotalStats({
        views: totalViews,
        likes: totalLikes,
        shares: totalShares,
        comments: totalComments
      })

      setContentTypeStats(
        Array.from(typeMap.entries()).map(([type, data]) => ({
          type,
          count: data.count,
          engagement: data.engagement
        }))
      )

      setTimeSeriesData(
        Array.from(dateMap.entries())
          .map(([date, data]) => ({
            date,
            ...data
          }))
          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
          .slice(-7) // Last 7 days
      )

    } catch (error) {
      console.error('Failed to load analytics:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  const COLORS = ['#8b5cf6', '#ec4899', '#10b981', '#f59e0b', '#3b82f6']

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <BarChart3 className="w-10 h-10 text-purple-400" />
            Analytics Dashboard
          </h1>
          <p className="text-slate-400">Track your content performance and engagement metrics</p>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400 mt-4">Loading analytics...</p>
        </div>
      ) : (
        <>
          {/* Stats Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <Eye className="w-6 h-6" />
                </div>
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-slate-400 text-sm mb-1">Total Views</h3>
              <p className="text-3xl font-bold">{formatNumber(totalStats.views)}</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-red-500 to-pink-500 flex items-center justify-center">
                  <Heart className="w-6 h-6" />
                </div>
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-slate-400 text-sm mb-1">Total Likes</h3>
              <p className="text-3xl font-bold">{formatNumber(totalStats.likes)}</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                  <Share2 className="w-6 h-6" />
                </div>
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-slate-400 text-sm mb-1">Total Shares</h3>
              <p className="text-3xl font-bold">{formatNumber(totalStats.shares)}</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center">
                  <MessageCircle className="w-6 h-6" />
                </div>
                <TrendingUp className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-slate-400 text-sm mb-1">Total Comments</h3>
              <p className="text-3xl font-bold">{formatNumber(totalStats.comments)}</p>
            </motion.div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Engagement Over Time */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center gap-3 mb-6">
                <Activity className="w-6 h-6 text-purple-400" />
                <h2 className="text-xl font-bold">Engagement Over Time</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="views" stroke="#8b5cf6" strokeWidth={2} />
                  <Line type="monotone" dataKey="likes" stroke="#ec4899" strokeWidth={2} />
                  <Line type="monotone" dataKey="shares" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Content Type Distribution */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="glass rounded-xl p-6"
            >
              <div className="flex items-center gap-3 mb-6">
                <PieChart className="w-6 h-6 text-purple-400" />
                <h2 className="text-xl font-bold">Content by Type</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <RechartsPie>
                  <Pie
                    data={contentTypeStats}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ type, count }) => `${type}: ${count}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {contentTypeStats.map((_entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                </RechartsPie>
              </ResponsiveContainer>
            </motion.div>

            {/* Engagement by Content Type */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="glass rounded-xl p-6 lg:col-span-2"
            >
              <div className="flex items-center gap-3 mb-6">
                <BarChart3 className="w-6 h-6 text-purple-400" />
                <h2 className="text-xl font-bold">Engagement by Content Type</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={contentTypeStats}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="type" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="engagement" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>
        </>
      )}
    </div>
  )
}
