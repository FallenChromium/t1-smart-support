"use client"

import { useState } from "react"
import { CheckIcon, CopyIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Item, ItemActions, ItemContent, ItemDescription, ItemHeader, ItemTitle } from "@/components/ui/item"
import { ConfidenceBadge } from "./confidence-badge"
import { cn } from "@/lib/utils"

interface RecommendationCardProps {
  title: string
  answer: string
  similarity: number
  index: number
  isSelected?: boolean
  onSelect?: () => void
}

export function RecommendationCard({
  title,
  answer,
  similarity,
  index,
  isSelected,
  onSelect,
}: RecommendationCardProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(answer)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Item
      className={cn(
        "cursor-pointer transition-all hover:border-primary/50",
        isSelected && "border-primary bg-primary/5",
      )}
      onClick={onSelect}
    >
      <ItemHeader>
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
              {index}
            </div>
            <ItemTitle className="text-balance">{title}</ItemTitle>
          </div>
          <ConfidenceBadge probability={similarity} />
        </div>
      </ItemHeader>
      <ItemContent>
        <ItemDescription className="line-clamp-3 text-pretty">{answer}</ItemDescription>
      </ItemContent>
      <ItemActions>
        <Button
          variant="outline"
          size="sm"
          onClick={(e) => {
            e.stopPropagation()
            handleCopy()
          }}
        >
          {copied ? (
            <>
              <CheckIcon className="h-4 w-4" />
              Скопировано
            </>
          ) : (
            <>
              <CopyIcon className="h-4 w-4" />
              Копировать
            </>
          )}
        </Button>
      </ItemActions>
    </Item>
  )
}
