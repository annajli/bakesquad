import { BookOpen, MessageSquare, SlidersHorizontal } from 'lucide-react'
import { useState } from 'react'
import { SidePane } from './SidePane'

interface Props {
  activeTab: 'chat' | 'box'
  onTabChange: (t: 'chat' | 'box') => void
  recency: string | null
  onRecencyChange: (v: string | null) => void
  children: React.ReactNode
}

export function AppLayout({
  activeTab,
  onTabChange,
  recency,
  onRecencyChange,
  children,
}: Props) {
  const [paneOpen, setPaneOpen] = useState(false)

  return (
    <div className="min-h-screen bg-[#fffbf5]">
      {/* Header */}
      <header className="sticky top-0 z-10 bg-white/90 backdrop-blur border-b border-amber-100 px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPaneOpen(true)}
            className="p-2 rounded-lg text-stone-500 hover:bg-amber-50 hover:text-amber-700 transition-colors"
            title="Settings"
          >
            <SlidersHorizontal size={20} />
          </button>
          <span className="text-xl font-bold tracking-tight text-amber-800">
            🧁 BakeSquad
          </span>
        </div>

        {/* Tab switcher */}
        <nav className="flex gap-1 bg-amber-50 rounded-xl p-1">
          <TabButton
            active={activeTab === 'chat'}
            onClick={() => onTabChange('chat')}
            icon={<MessageSquare size={15} />}
            label="Find Recipes"
          />
          <TabButton
            active={activeTab === 'box'}
            onClick={() => onTabChange('box')}
            icon={<BookOpen size={15} />}
            label="Recipe Box"
          />
        </nav>

        {/* Spacer to balance the left settings button */}
        <div className="w-10" />
      </header>

      <SidePane
        open={paneOpen}
        onClose={() => setPaneOpen(false)}
        recency={recency}
        onRecencyChange={onRecencyChange}
      />

      <main className="max-w-5xl mx-auto px-4 py-6">{children}</main>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all
        ${active
          ? 'bg-white text-amber-800 shadow-sm'
          : 'text-stone-500 hover:text-stone-700'
        }`}
    >
      {icon}
      {label}
    </button>
  )
}
