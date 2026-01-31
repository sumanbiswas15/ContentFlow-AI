import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { User, Key, Bell, Shield, Save, AlertCircle } from 'lucide-react'
import api from '../lib/api'
import { useAuthStore } from '../store/authStore'

export default function Settings() {
  const { user, updateUser } = useAuthStore()
  const [activeTab, setActiveTab] = useState('profile')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  const [profileData, setProfileData] = useState({
    username: user?.username || '',
    email: user?.email || '',
    full_name: user?.full_name || '',
  })

  const [usageData, setUsageData] = useState<any>(null)

  useEffect(() => {
    if (activeTab === 'usage') {
      loadUsageData()
    }
  }, [activeTab])

  const loadUsageData = async () => {
    try {
      const response = await api.get('/auth/usage')
      setUsageData(response.data)
    } catch (error) {
      console.error('Failed to load usage data:', error)
    }
  }

  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setMessage(null)

    try {
      const response = await api.put('/auth/profile', profileData)
      updateUser(response.data.user)
      setMessage({ type: 'success', text: 'Profile updated successfully' })
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || 'Failed to update profile' })
    } finally {
      setLoading(false)
    }
  }

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'api-keys', label: 'API Keys', icon: Key },
    { id: 'usage', label: 'Usage & Limits', icon: Shield },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ]

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2">Settings</h1>
        <p className="text-slate-400">Manage your account and preferences</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Tabs */}
        <div className="lg:col-span-1">
          <div className="glass rounded-xl p-4 space-y-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all ${
                  activeTab === tab.id
                    ? 'bg-gradient-to-r from-purple-500/20 to-indigo-500/20 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="lg:col-span-3">
          <div className="glass rounded-xl p-6">
            {message && (
              <div className={`mb-6 p-4 rounded-lg flex items-start space-x-3 ${
                message.type === 'success'
                  ? 'bg-green-500/10 border border-green-500/50'
                  : 'bg-red-500/10 border border-red-500/50'
              }`}>
                <AlertCircle className={`w-5 h-5 flex-shrink-0 mt-0.5 ${
                  message.type === 'success' ? 'text-green-500' : 'text-red-500'
                }`} />
                <p className={`text-sm ${
                  message.type === 'success' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {message.text}
                </p>
              </div>
            )}

            {activeTab === 'profile' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold mb-6">Profile Settings</h2>
                <form onSubmit={handleProfileUpdate} className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium mb-2">Username</label>
                    <input
                      type="text"
                      value={profileData.username}
                      onChange={(e) => setProfileData({ ...profileData, username: e.target.value })}
                      className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                      disabled
                    />
                    <p className="text-xs text-slate-500 mt-1">Username cannot be changed</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Email</label>
                    <input
                      type="email"
                      value={profileData.email}
                      onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
                      className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium mb-2">Full Name</label>
                    <input
                      type="text"
                      value={profileData.full_name}
                      onChange={(e) => setProfileData({ ...profileData, full_name: e.target.value })}
                      className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Save className="w-5 h-5" />
                    <span>{loading ? 'Saving...' : 'Save Changes'}</span>
                  </button>
                </form>
              </motion.div>
            )}

            {activeTab === 'api-keys' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold mb-6">API Keys</h2>
                <p className="text-slate-400 mb-6">
                  Manage your API keys for programmatic access to ContentFlow AI
                </p>
                <button className="flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all">
                  <Key className="w-5 h-5" />
                  <span>Create New API Key</span>
                </button>
                <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
                  <p className="text-sm text-slate-400">No API keys created yet</p>
                </div>
              </motion.div>
            )}

            {activeTab === 'usage' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold mb-6">Usage & Limits</h2>
                {usageData ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="font-semibold mb-4">Current Usage</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <UsageCard
                          label="Tokens Used (Month)"
                          value={usageData.usage_stats.tokens_used_this_month.toLocaleString()}
                          limit={usageData.usage_limits.daily_token_limit.toLocaleString()}
                        />
                        <UsageCard
                          label="Cost (Month)"
                          value={`$${usageData.usage_stats.cost_this_month.toFixed(2)}`}
                          limit={`$${usageData.usage_limits.monthly_cost_limit.toFixed(2)}`}
                        />
                        <UsageCard
                          label="Content Items"
                          value={usageData.usage_stats.content_items_created.toString()}
                          limit={usageData.usage_limits.max_content_items.toString()}
                        />
                        <UsageCard
                          label="Storage Used"
                          value={`${usageData.usage_stats.storage_used_mb} MB`}
                          limit={`${usageData.usage_limits.max_storage_mb} MB`}
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
                  </div>
                )}
              </motion.div>
            )}

            {activeTab === 'notifications' && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <h2 className="text-2xl font-bold mb-6">Notification Preferences</h2>
                <div className="space-y-4">
                  <NotificationToggle
                    label="Job Completion"
                    description="Get notified when your AI jobs complete"
                  />
                  <NotificationToggle
                    label="Usage Alerts"
                    description="Receive alerts when approaching usage limits"
                  />
                  <NotificationToggle
                    label="Product Updates"
                    description="Stay informed about new features and improvements"
                  />
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function UsageCard({ label, value, limit }: { label: string; value: string; limit: string }) {
  return (
    <div className="p-4 bg-slate-900/30 rounded-lg">
      <p className="text-sm text-slate-400 mb-1">{label}</p>
      <p className="text-2xl font-bold mb-1">{value}</p>
      <p className="text-xs text-slate-500">Limit: {limit}</p>
    </div>
  )
}

function NotificationToggle({ label, description }: { label: string; description: string }) {
  const [enabled, setEnabled] = useState(true)

  return (
    <div className="flex items-center justify-between p-4 bg-slate-900/30 rounded-lg">
      <div>
        <p className="font-medium">{label}</p>
        <p className="text-sm text-slate-400">{description}</p>
      </div>
      <button
        onClick={() => setEnabled(!enabled)}
        className={`relative w-12 h-6 rounded-full transition-colors ${
          enabled ? 'bg-purple-500' : 'bg-slate-700'
        }`}
      >
        <div
          className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
            enabled ? 'translate-x-6' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  )
}
