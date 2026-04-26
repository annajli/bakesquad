import { useState } from 'react'
import { AppLayout } from './components/layout/AppLayout'
import { ChatTab } from './components/chat/ChatTab'
import { RecipeBoxTab } from './components/recipebox/RecipeBoxTab'

type Tab = 'chat' | 'box'

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  const [recency, setRecency] = useState<string | null>(null)

  return (
    <AppLayout
      activeTab={tab}
      onTabChange={setTab}
      recency={recency}
      onRecencyChange={setRecency}
    >
      {/* Both tabs stay mounted — CSS hides the inactive one to preserve state */}
      <div className={tab === 'chat' ? '' : 'hidden'}>
        <ChatTab recency={recency} />
      </div>
      <div className={tab === 'box' ? '' : 'hidden'}>
        <RecipeBoxTab />
      </div>
    </AppLayout>
  )
}
