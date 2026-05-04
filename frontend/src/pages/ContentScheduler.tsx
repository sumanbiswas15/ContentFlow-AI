import { useState, useEffect, useCallback } from 'react'
import { Calendar, dateFnsLocalizer } from 'react-big-calendar'
import { format, parse, startOfWeek, getDay } from 'date-fns'
import { enUS } from 'date-fns/locale'
import { Calendar as CalendarIcon, Plus, Send, Trash2 } from 'lucide-react'
import api from '../lib/api'
import 'react-big-calendar/lib/css/react-big-calendar.css'

const locales = {
  'en-US': enUS
}

const localizer = dateFnsLocalizer({
  format,
  parse,
  startOfWeek,
  getDay,
  locales,
})

interface ScheduledPost {
  id: string
  title: string
  content: string
  scheduled_time: Date
  platform: string
  status: 'pending' | 'published' | 'failed'
}

export default function ContentScheduler() {
  const [events, setEvents] = useState<any[]>([])
  const [selectedEvent, setSelectedEvent] = useState<ScheduledPost | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadScheduledPosts()
  }, [])

  const loadScheduledPosts = async () => {
    try {
      setLoading(true)
      const response = await api.get('/jobs/', {  // Ensure trailing slash
        params: {
          limit: 100
        }
      })
      
      // Filter for jobs that have scheduled_time in parameters
      const scheduledEvents = (response.data || [])
        .filter((job: any) => job.parameters?.scheduled_time)
        .map((job: any) => ({
          id: job.job_id,
          title: job.parameters?.title || 'Scheduled Post',
          start: new Date(job.parameters.scheduled_time),
          end: new Date(job.parameters.scheduled_time),
          resource: {
            id: job.job_id,
            title: job.parameters?.title || '',
            content: job.parameters?.content || '',
            scheduled_time: new Date(job.parameters.scheduled_time),
            platform: job.parameters?.platform || 'twitter',
            status: job.status
          }
        }))
      
      setEvents(scheduledEvents)
    } catch (error) {
      console.error('Failed to load scheduled posts:', error)
      setEvents([])
    } finally {
      setLoading(false)
    }
  }

  const handleSelectSlot = useCallback(({ start }: { start: Date }) => {
    setSelectedEvent({
      id: '',
      title: '',
      content: '',
      scheduled_time: start,
      platform: 'twitter',
      status: 'pending'
    })
    setShowModal(true)
  }, [])

  const handleSelectEvent = useCallback((event: any) => {
    setSelectedEvent(event.resource)
    setShowModal(true)
  }, [])

  const handleSaveSchedule = async () => {
    if (!selectedEvent) return

    try {
      if (selectedEvent.id) {
        // Update existing - Note: Update endpoint may not exist yet
        // For now, delete and recreate
        await api.delete(`/jobs/${selectedEvent.id}`)
      }
      
      // Create new scheduled job
      const response = await api.post('/jobs/submit', {
        job_type: 'content_generation',
        engine: 'social_media_planner',
        operation: 'schedule_post',
        parameters: {
          title: selectedEvent.title,
          content: selectedEvent.content,
          platform: selectedEvent.platform,
          scheduled_time: selectedEvent.scheduled_time.toISOString()
        },
        priority: 5
      })
      
      console.log('Job scheduled:', response.data)
      setShowModal(false)
      loadScheduledPosts()
    } catch (error) {
      console.error('Failed to save schedule:', error)
      alert('Failed to schedule post. Please try again.')
    }
  }

  const handleDeleteSchedule = async () => {
    if (!selectedEvent?.id) return

    try {
      await api.delete(`/jobs/${selectedEvent.id}`)
      setShowModal(false)
      loadScheduledPosts()
    } catch (error) {
      console.error('Failed to delete schedule:', error)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <CalendarIcon className="w-10 h-10 text-purple-400" />
            Content Scheduler
          </h1>
          <p className="text-slate-400">Schedule your content for optimal engagement</p>
        </div>
        <button
          onClick={() => {
            setSelectedEvent({
              id: '',
              title: '',
              content: '',
              scheduled_time: new Date(),
              platform: 'twitter',
              status: 'pending'
            })
            setShowModal(true)
          }}
          className="flex items-center gap-2 px-6 py-3 bg-purple-500 hover:bg-purple-600 rounded-lg font-medium transition-all"
        >
          <Plus className="w-5 h-5" />
          Schedule Post
        </button>
      </div>

      {/* Calendar */}
      <div className="glass rounded-xl p-6">
        {loading ? (
          <div className="text-center py-12">
            <div className="inline-block w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-slate-400 mt-4">Loading schedule...</p>
          </div>
        ) : (
          <div style={{ height: '600px' }} className="calendar-container">
            <Calendar
              localizer={localizer}
              events={events}
              startAccessor="start"
              endAccessor="end"
              onSelectSlot={handleSelectSlot}
              onSelectEvent={handleSelectEvent}
              selectable
              style={{ height: '100%' }}
            />
          </div>
        )}
      </div>

      {/* Schedule Modal */}
      {showModal && selectedEvent && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="glass rounded-2xl max-w-2xl w-full p-6">
            <h2 className="text-2xl font-bold mb-6">
              {selectedEvent.id ? 'Edit' : 'Schedule'} Post
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Title</label>
                <input
                  type="text"
                  value={selectedEvent.title}
                  onChange={(e) => setSelectedEvent({ ...selectedEvent, title: e.target.value })}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Post title..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Content</label>
                <textarea
                  value={selectedEvent.content}
                  onChange={(e) => setSelectedEvent({ ...selectedEvent, content: e.target.value })}
                  rows={4}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder="Post content..."
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Platform</label>
                  <select
                    value={selectedEvent.platform}
                    onChange={(e) => setSelectedEvent({ ...selectedEvent, platform: e.target.value })}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  >
                    <option value="twitter">Twitter</option>
                    <option value="linkedin">LinkedIn</option>
                    <option value="facebook">Facebook</option>
                    <option value="instagram">Instagram</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Scheduled Time</label>
                  <input
                    type="datetime-local"
                    value={format(selectedEvent.scheduled_time, "yyyy-MM-dd'T'HH:mm")}
                    onChange={(e) => setSelectedEvent({ ...selectedEvent, scheduled_time: new Date(e.target.value) })}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3 mt-6">
              <button
                onClick={handleSaveSchedule}
                className="flex-1 flex items-center justify-center gap-2 py-3 bg-purple-500 hover:bg-purple-600 rounded-lg font-medium transition-all"
              >
                <Send className="w-5 h-5" />
                {selectedEvent.id ? 'Update' : 'Schedule'}
              </button>
              
              {selectedEvent.id && (
                <button
                  onClick={handleDeleteSchedule}
                  className="px-6 py-3 bg-red-500/20 hover:bg-red-500/30 text-red-400 rounded-lg font-medium transition-all"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              )}
              
              <button
                onClick={() => setShowModal(false)}
                className="px-6 py-3 glass hover:bg-slate-800 rounded-lg font-medium transition-all"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .calendar-container .rbc-calendar {
          background: transparent;
          color: #e2e8f0;
        }
        .calendar-container .rbc-header {
          background: rgba(15, 23, 42, 0.5);
          border-color: #334155;
          padding: 12px;
          font-weight: 600;
        }
        .calendar-container .rbc-today {
          background-color: rgba(168, 85, 247, 0.1);
        }
        .calendar-container .rbc-event {
          background-color: #a855f7;
          border: none;
          border-radius: 4px;
        }
        .calendar-container .rbc-day-bg,
        .calendar-container .rbc-month-view {
          border-color: #334155;
        }
        .calendar-container .rbc-off-range-bg {
          background: rgba(15, 23, 42, 0.3);
        }
      `}</style>
    </div>
  )
}
