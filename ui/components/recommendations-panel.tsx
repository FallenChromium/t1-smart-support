"use client"

import { SparklesIcon, CopyIcon, CheckIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { Kbd } from "@/components/ui/kbd"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { ConfidenceBadge } from "@/components/confidence-badge"
import { cn } from "@/lib/utils"
import type { Recommendation } from "@/types/chat"
import { useState } from "react"

interface RecommendationsPanelProps {
  recommendations: Recommendation[]
  isLoading: boolean
  selectedIndex: number | null
  onSelect: (index: number) => void
  onInsert: (answer: string) => void
  category?: string
}

export function RecommendationsPanel({
  recommendations,
  isLoading,
  selectedIndex,
  onSelect,
  onInsert,
  category,
}: RecommendationsPanelProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const handleCopy = async (rec: Recommendation) => {
    await navigator.clipboard.writeText(rec.answer)
    setCopiedId(rec.id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  return (
    <div className="flex h-full flex-col border-l bg-muted/20">
      {/* Header */}
      <div className="border-b bg-background p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <SparklesIcon className="h-5 w-5 text-primary" />
            <h2 className="font-semibold">AI Рекомендации</h2>
          </div>
          {recommendations.length > 0 && (
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Kbd>Alt</Kbd>+<Kbd>1-3</Kbd>
            </div>
          )}
        </div>
        {category && (
          <div className="mt-2 text-xs text-muted-foreground">
            Категория: <span className="font-medium text-foreground">{category}</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <Spinner className="mx-auto mb-3 h-8 w-8" />
              <p className="text-sm text-muted-foreground">Анализируем сообщение...</p>
            </div>
          </div>
        ) : recommendations.length > 0 ? (
          <div className="space-y-3">
            {recommendations.map((rec, index) => (
              <button
                key={rec.id}
                onClick={() => onSelect(index)}
                className={cn(
                  "w-full rounded-lg border bg-card p-3 text-left transition-all hover:shadow-md",
                  selectedIndex === index && "ring-2 ring-primary ring-offset-2",
                )}
              >
                <div className="mb-2 flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="flex h-5 w-5 items-center justify-center rounded bg-primary/10 text-xs font-bold text-primary">
                      {index + 1}
                    </span>
                    <ConfidenceBadge probability={rec.similarity} />
                  </div>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-6 w-6 shrink-0"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleCopy(rec)
                    }}
                  >
                    {copiedId === rec.id ? <CheckIcon className="h-3 w-3" /> : <CopyIcon className="h-3 w-3" />}
                  </Button>
                </div>

                <p className="mb-2 line-clamp-2 text-xs font-medium text-muted-foreground">{rec.query}</p>
                <p className="line-clamp-4 text-sm leading-relaxed">{rec.answer}</p>

                {selectedIndex === index && (
                  <Button
                    size="sm"
                    className="mt-3 w-full"
                    onClick={(e) => {
                      e.stopPropagation()
                      onInsert(rec.answer)
                    }}
                  >
                    Вставить в ответ
                    <Kbd className="ml-2">Ctrl</Kbd>+<Kbd>Shift</Kbd>+<Kbd>I</Kbd>
                  </Button>
                )}
              </button>
            ))}
          </div>
        ) : (
          <Empty className="h-full">
            <EmptyHeader>
              <EmptyMedia>
                <SparklesIcon className="h-10 w-10" />
              </EmptyMedia>
              <EmptyTitle>Ожидание сообщения</EmptyTitle>
              <EmptyDescription>
                AI-рекомендации появятся автоматически при получении сообщения от клиента
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
      </div>
    </div>
  )
}
