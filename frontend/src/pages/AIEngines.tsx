import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import { 
  MessageSquare, Image, Music, Video, Sparkles, 
  TrendingUp, Zap, ArrowRight, Check 
} from 'lucide-react'

export default function AIEngines() {
  const navigate = useNavigate()

  const engines = [
    {
      id: 'text',
      name: 'Text Intelligence',
      icon: MessageSquare,
      color: 'from-blue-500 to-cyan-500',
      description: 'Generate, summarize, and transform text content with advanced AI',
      features: [
        'Content generation (blogs, captions, scripts)',
        'Text summarization',
        'Tone transformation',
        'Multi-language translation',
        'Platform adaptation',
      ],
      route: '/app/engines/text',
    },
    {
      id: 'image',
      name: 'Image Generation',
      icon: Image,
      color: 'from-purple-500 to-pink-500',
      description: 'Create stunning visuals for any platform or purpose',
      features: [
        'Thumbnails & posters',
        'Social media graphics',
        'Custom dimensions',
        'Style customization',
        'High-quality output',
      ],
      route: '/app/engines/image',
    },
    {
      id: 'audio',
      name: 'Audio Generation',
      icon: Music,
      color: 'from-green-500 to-emerald-500',
      description: 'Generate music, voiceovers, and narrations',
      features: [
        'Background music',
        'Voiceovers',
        'Narrations',
        'Multiple formats',
        'Quality control',
      ],
      route: '/app/engines/audio',
    },
    {
      id: 'video',
      name: 'Video Pipeline',
      icon: Video,
      color: 'from-orange-500 to-red-500',
      description: 'Create and edit videos with AI-powered tools',
      features: [
        'Short-form videos',
        'Explainer videos',
        'Automated editing',
        'Multiple resolutions',
        'Format conversion',
      ],
      route: '/app/engines/video',
    },
    {
      id: 'creative',
      name: 'Creative Assistant',
      icon: Sparkles,
      color: 'from-yellow-500 to-amber-500',
      description: 'Get creative suggestions and iterative refinement',
      features: [
        'Idea generation',
        'Creative suggestions',
        'Design assistance',
        'Marketing help',
        'Iterative refinement',
      ],
      route: '/app/engines/creative',
    },
    {
      id: 'social',
      name: 'Social Media Planner',
      icon: TrendingUp,
      color: 'from-indigo-500 to-purple-500',
      description: 'Optimize content for social media platforms',
      features: [
        'Platform optimization',
        'Hashtag generation',
        'CTA creation',
        'Posting time suggestions',
        'Engagement prediction',
      ],
      route: '/app/engines/social',
    },
    {
      id: 'analytics',
      name: 'Discovery Analytics',
      icon: Zap,
      color: 'from-pink-500 to-rose-500',
      description: 'Analyze and improve your content performance',
      features: [
        'Auto-tagging',
        'Trend analysis',
        'Engagement analysis',
        'Improvement suggestions',
        'Performance insights',
      ],
      route: '/app/engines/analytics',
    },
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2">
          AI Engines <span className="text-gradient">✨</span>
        </h1>
        <p className="text-slate-400">
          Powerful AI tools to supercharge your content creation
        </p>
      </div>

      {/* Engines Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {engines.map((engine, index) => (
          <motion.div
            key={engine.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass rounded-xl p-6 glass-hover cursor-pointer"
          >
            <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${engine.color} flex items-center justify-center mb-4`}>
              <engine.icon className="w-8 h-8 text-white" />
            </div>

            <h3 className="text-xl font-bold mb-2">{engine.name}</h3>
            <p className="text-slate-400 text-sm mb-4">{engine.description}</p>

            <div className="space-y-2 mb-4">
              {engine.features.slice(0, 3).map((feature, i) => (
                <div key={i} className="flex items-center space-x-2 text-sm">
                  <Check className="w-4 h-4 text-green-500 flex-shrink-0" />
                  <span className="text-slate-300">{feature}</span>
                </div>
              ))}
              {engine.features.length > 3 && (
                <p className="text-xs text-slate-500 pl-6">
                  +{engine.features.length - 3} more features
                </p>
              )}
            </div>

            <button 
              onClick={() => navigate(engine.route)}
              className="w-full flex items-center justify-center space-x-2 py-2 bg-slate-800/50 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <span className="text-sm font-medium">Try Engine</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </motion.div>
        ))}
      </div>

      {/* Orchestrator Section */}
      <div className="glass rounded-xl p-8">
        <div className="flex items-start space-x-6">
          <div className="w-20 h-20 rounded-xl bg-gradient-to-br from-violet-500 to-purple-500 flex items-center justify-center flex-shrink-0">
            <Sparkles className="w-10 h-10 text-white" />
          </div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold mb-2">AI Orchestrator</h2>
            <p className="text-slate-400 mb-4">
              Combine multiple AI engines into powerful workflows. The orchestrator
              intelligently coordinates engines to handle complex content creation tasks.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              <div className="flex items-center space-x-2 text-sm">
                <Check className="w-4 h-4 text-green-500" />
                <span>Multi-engine workflows</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <Check className="w-4 h-4 text-green-500" />
                <span>Intelligent routing</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <Check className="w-4 h-4 text-green-500" />
                <span>Error handling</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <Check className="w-4 h-4 text-green-500" />
                <span>Progress tracking</span>
              </div>
            </div>
            <button 
              onClick={() => navigate('/app/engines/orchestrator')}
              className="flex items-center space-x-2 px-6 py-3 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all"
            >
              <span>Create Workflow</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Usage Tips */}
      <div className="glass rounded-xl p-6">
        <h2 className="text-xl font-bold mb-4">💡 Usage Tips</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-slate-900/30 rounded-lg">
            <h3 className="font-semibold mb-2">Start Simple</h3>
            <p className="text-sm text-slate-400">
              Begin with single engines to understand their capabilities before
              creating complex workflows.
            </p>
          </div>
          <div className="p-4 bg-slate-900/30 rounded-lg">
            <h3 className="font-semibold mb-2">Monitor Usage</h3>
            <p className="text-sm text-slate-400">
              Keep track of your token usage and costs in the dashboard to
              optimize your spending.
            </p>
          </div>
          <div className="p-4 bg-slate-900/30 rounded-lg">
            <h3 className="font-semibold mb-2">Iterate & Refine</h3>
            <p className="text-sm text-slate-400">
              Use the Creative Assistant for iterative refinement to get the
              best results.
            </p>
          </div>
          <div className="p-4 bg-slate-900/30 rounded-lg">
            <h3 className="font-semibold mb-2">Combine Engines</h3>
            <p className="text-sm text-slate-400">
              Use the Orchestrator to combine multiple engines for end-to-end
              content creation.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
