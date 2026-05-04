import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Sparkles, Zap, Image, Music, Video, MessageSquare, 
  TrendingUp, ArrowRight, Check 
} from 'lucide-react'

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }} />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-8 h-8 text-purple-400" />
            <span className="text-2xl font-bold text-gradient">ContentFlow AI</span>
          </div>
          <div className="flex items-center space-x-4">
            <Link to="/login" className="px-4 py-2 text-slate-300 hover:text-white transition-colors">
              Login
            </Link>
            <Link to="/register" className="px-6 py-2 gradient-primary rounded-lg font-medium hover:shadow-lg hover:shadow-purple-500/50 transition-all">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-6xl md:text-7xl font-bold mb-6">
              Create Content with
              <span className="block text-gradient">AI Magic ✨</span>
            </h1>
            <p className="text-xl text-slate-400 mb-8 max-w-2xl mx-auto">
              Transform your ideas into engaging content across all platforms with 7 powerful AI engines
            </p>
            <div className="flex items-center justify-center space-x-4">
              <Link to="/register" className="px-8 py-4 gradient-primary rounded-lg font-medium text-lg hover:shadow-xl hover:shadow-purple-500/50 transition-all flex items-center space-x-2">
                <span>Start Creating</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <button className="px-8 py-4 glass rounded-lg font-medium text-lg glass-hover">
                Watch Demo
              </button>
            </div>
          </motion.div>

          {/* Feature Cards */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-20"
          >
            <FeatureCard
              icon={<Zap className="w-8 h-8" />}
              title="Lightning Fast"
              description="Generate content in seconds with our optimized AI engines"
            />
            <FeatureCard
              icon={<Sparkles className="w-8 h-8" />}
              title="7 AI Engines"
              description="Text, Image, Audio, Video, and more - all in one platform"
            />
            <FeatureCard
              icon={<TrendingUp className="w-8 h-8" />}
              title="Optimize & Analyze"
              description="Get insights and optimize content for maximum engagement"
            />
          </motion.div>
        </div>
      </section>

      {/* AI Engines Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">
            Powered by <span className="text-gradient">7 AI Engines</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <EngineCard icon={<MessageSquare />} title="Text Intelligence" color="from-blue-500 to-cyan-500" />
            <EngineCard icon={<Image />} title="Image Generation" color="from-purple-500 to-pink-500" />
            <EngineCard icon={<Music />} title="Audio Generation" color="from-green-500 to-emerald-500" />
            <EngineCard icon={<Video />} title="Video Pipeline" color="from-orange-500 to-red-500" />
            <EngineCard icon={<Sparkles />} title="Creative Assistant" color="from-yellow-500 to-amber-500" />
            <EngineCard icon={<TrendingUp />} title="Social Media Planner" color="from-indigo-500 to-purple-500" />
            <EngineCard icon={<Zap />} title="Discovery Analytics" color="from-pink-500 to-rose-500" />
            <EngineCard icon={<Sparkles />} title="AI Orchestrator" color="from-violet-500 to-purple-500" />
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12">
            Simple, <span className="text-gradient">Transparent Pricing</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <PricingCard
              name="Starter"
              price="$5"
              features={[
                '10,000 tokens/month',
                'All 7 AI engines',
                'Basic analytics',
                'Email support'
              ]}
            />
            <PricingCard
              name="Pro"
              price="$10"
              features={[
                '100,000 tokens/month',
                'All 7 AI engines',
                'Advanced analytics',
                'Priority support',
                'Custom workflows'
              ]}
              highlighted
            />
            <PricingCard
              name="Enterprise"
              price="Custom"
              features={[
                'Unlimited tokens',
                'All 7 AI engines',
                'Enterprise analytics',
                '24/7 support',
                'Custom integrations',
                'Dedicated account manager'
              ]}
            />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative z-10 px-6 py-20">
        <div className="max-w-4xl mx-auto text-center glass rounded-2xl p-12">
          <h2 className="text-4xl font-bold mb-4">
            Ready to Transform Your Content?
          </h2>
          <p className="text-xl text-slate-400 mb-8">
            Join thousands of creators using AI to supercharge their content
          </p>
          <Link to="/register" className="inline-flex items-center space-x-2 px-8 py-4 gradient-primary rounded-lg font-medium text-lg hover:shadow-xl hover:shadow-purple-500/50 transition-all">
            <span>Get Started Free</span>
            <ArrowRight className="w-5 h-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 px-6 py-8 border-t border-slate-800">
        <div className="max-w-7xl mx-auto text-center text-slate-500">
          <p>&copy; 2024 ContentFlow AI. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="glass rounded-xl p-6 glass-hover"
    >
      <div className="w-12 h-12 gradient-primary rounded-lg flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold mb-2">{title}</h3>
      <p className="text-slate-400">{description}</p>
    </motion.div>
  )
}

function EngineCard({ icon, title, color }: { icon: React.ReactNode; title: string; color: string }) {
  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      className="glass rounded-xl p-6 glass-hover cursor-pointer"
    >
      <div className={`w-12 h-12 bg-gradient-to-br ${color} rounded-lg flex items-center justify-center mb-4`}>
        {icon}
      </div>
      <h3 className="font-semibold">{title}</h3>
    </motion.div>
  )
}

function PricingCard({ name, price, features, highlighted }: { name: string; price: string; features: string[]; highlighted?: boolean }) {
  return (
    <div className={`glass rounded-xl p-8 ${highlighted ? 'ring-2 ring-purple-500' : ''}`}>
      {highlighted && (
        <div className="text-center mb-4">
          <span className="px-3 py-1 gradient-primary rounded-full text-sm font-medium">Most Popular</span>
        </div>
      )}
      <h3 className="text-2xl font-bold mb-2">{name}</h3>
      <div className="mb-6">
        <span className="text-4xl font-bold">{price}</span>
        {price !== 'Custom' && <span className="text-slate-400">/month</span>}
      </div>
      <ul className="space-y-3 mb-8">
        {features.map((feature, i) => (
          <li key={i} className="flex items-center space-x-2">
            <Check className="w-5 h-5 text-green-500" />
            <span className="text-slate-300">{feature}</span>
          </li>
        ))}
      </ul>
      <button className={`w-full py-3 rounded-lg font-medium transition-all ${
        highlighted 
          ? 'gradient-primary hover:shadow-lg hover:shadow-purple-500/50' 
          : 'glass glass-hover'
      }`}>
        Get Started
      </button>
    </div>
  )
}
