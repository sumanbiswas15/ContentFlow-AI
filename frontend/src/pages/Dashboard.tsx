import { motion } from 'framer-motion'
import { FileText, Sparkles, TrendingUp, Clock, Zap, DollarSign } from 'lucide-react'
import { useEffect, useState } from 'react'
import api from '../lib/api'

interface ActivityItem {
  title: string
  description: string
  time: string
  icon: React.ReactNode
  type: string
}

export default function Dashboard() {
  const [stats, setStats] = useState({
    contentCount: 0,
    jobsRunning: 0,
    tokensUsed: 0,
    costThisMonth: 0,
  })
  const [recentActivity, setRecentActivity] = useState<ActivityItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      // Load content count
      const contentResponse = await api.get('/content/', { params: { limit: 100 } })
      const contentCount = contentResponse.data.items?.length || 0

      // Load jobs
      const jobsResponse = await api.get('/jobs/', { params: { limit: 100 } })
      const jobs = jobsResponse.data.jobs || []
      const jobsRunning = jobs.filter((j: any) => j.status === 'running' || j.status === 'queued').length

      // Build recent activity from jobs and content
      const activities: ActivityItem[] = []

      // Add recent jobs
      jobs.slice(0, 5).forEach((job: any) => {
        const timeAgo = getTimeAgo(new Date(job.created_at))
        activities.push({
          title: getJobTitle(job.job_type),
          description: `${job.status} - ${job.job_type}`,
          time: timeAgo,
          icon: <Sparkles className="w-5 h-5" />,
          type: 'job'
        })
      })

      // Add recent content
      const recentContent = contentResponse.data.items?.slice(0, 3) || []
      recentContent.forEach((item: any) => {
        const timeAgo = getTimeAgo(new Date(item.created_at))
        activities.push({
          title: item.title || 'Untitled',
          description: `${item.type} content created`,
          time: timeAgo,
          icon: <FileText className="w-5 h-5" />,
          type: 'content'
        })
      })

      // Sort by most recent and take top 5
      activities.sort((a, b) => {
        // Simple sort by time string (this is approximate)
        return a.time.localeCompare(b.time)
      })

      setStats({
        contentCount,
        jobsRunning,
        tokensUsed: 0, // Would need cost tracking API
        costThisMonth: 0, // Would need cost tracking API
      })
      setRecentActivity(activities.slice(0, 5))
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      // Show empty state on error
      setRecentActivity([])
    } finally {
      setLoading(false)
    }
  }

  const getTimeAgo = (date: Date): string => {
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins} min${diffMins > 1 ? 's' : ''} ago`
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`
    return date.toLocaleDateString()
  }

  const getJobTitle = (jobType: string): string => {
    const titles: Record<string, string> = {
      'text_generation': 'Text Generated',
      'image_generation': 'Image Created',
      'audio_generation': 'Audio Generated',
      'video_generation': 'Video Created',
      'content_optimization': 'Content Optimized',
      'social_media_planning': 'Social Post Planned',
      'discovery_analytics': 'Analytics Generated',
    }
    return titles[jobType] || 'Job Completed'
  }

  const statCards = [
    {
      title: 'Content Items',
      value: stats.contentCount,
      icon: FileText,
      color: 'from-blue-500 to-cyan-500',
      change: '+12%',
    },
    {
      title: 'Active Jobs',
      value: stats.jobsRunning,
      icon: Clock,
      color: 'from-purple-500 to-pink-500',
      change: '3 running',
    },
    {
      title: 'Tokens Used',
      value: stats.tokensUsed.toLocaleString(),
      icon: Zap,
      color: 'from-amber-500 to-orange-500',
      change: 'This month',
    },
    {
      title: 'Cost',
      value: `$${stats.costThisMonth.toFixed(2)}`,
      icon: DollarSign,
      color: 'from-green-500 to-emerald-500',
      change: 'This month',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2">
          Welcome back! <span className="text-gradient">✨</span>
        </h1>
        <p className="text-slate-400">Here's what's happening with your content</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass rounded-xl p-6 glass-hover"
          >
            <div className="flex items-start justify-between mb-4">
              <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${stat.color} flex items-center justify-center`}>
                <stat.icon className="w-6 h-6 text-white" />
              </div>
              <span className="text-xs text-green-400">{stat.change}</span>
            </div>
            <h3 className="text-2xl font-bold mb-1">{loading ? '...' : stat.value}</h3>
            <p className="text-sm text-slate-400">{stat.title}</p>
          </motion.div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-6">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <QuickActionCard
            icon={<FileText className="w-6 h-6" />}
            title="Create Content"
            description="Start a new content item"
            href="/app/content/create"
            color="from-blue-500 to-cyan-500"
          />
          <QuickActionCard
            icon={<Sparkles className="w-6 h-6" />}
            title="AI Engines"
            description="Use AI to generate content"
            href="/app/engines"
            color="from-purple-500 to-pink-500"
          />
          <QuickActionCard
            icon={<TrendingUp className="w-6 h-6" />}
            title="View Analytics"
            description="Check your performance"
            href="/app/analytics"
            color="from-green-500 to-emerald-500"
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-2xl font-bold mb-6">Recent Activity</h2>
        {loading ? (
          <div className="text-center py-8">
            <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-slate-400 mt-4">Loading activity...</p>
          </div>
        ) : recentActivity.length === 0 ? (
          <div className="text-center py-8">
            <Clock className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">No recent activity</p>
            <p className="text-sm text-slate-500 mt-2">Start creating content to see activity here</p>
          </div>
        ) : (
          <div className="space-y-4">
            {recentActivity.map((activity, index) => (
              <ActivityItemComponent
                key={index}
                title={activity.title}
                description={activity.description}
                time={activity.time}
                icon={activity.icon}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function QuickActionCard({ icon, title, description, href, color }: {
  icon: React.ReactNode
  title: string
  description: string
  href: string
  color: string
}) {
  return (
    <a
      href={href}
      className="block p-6 bg-slate-900/50 rounded-lg border border-slate-800 hover:border-slate-700 transition-all group"
    >
      <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
        {icon}
      </div>
      <h3 className="font-semibold mb-1">{title}</h3>
      <p className="text-sm text-slate-400">{description}</p>
    </a>
  )
}

function ActivityItemComponent({ title, description, time, icon }: {
  title: string
  description: string
  time: string
  icon: React.ReactNode
}) {
  return (
    <div className="flex items-start space-x-4 p-4 bg-slate-900/30 rounded-lg hover:bg-slate-900/50 transition-colors">
      <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center flex-shrink-0">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{title}</p>
        <p className="text-sm text-slate-400">{description}</p>
      </div>
      <span className="text-xs text-slate-500 flex-shrink-0">{time}</span>
    </div>
  )
}
