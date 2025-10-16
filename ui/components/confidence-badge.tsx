import { Badge } from "@/components/ui/badge"
import { getConfidenceInfo, formatProbability } from "@/lib/confidence-utils"
import { cn } from "@/lib/utils"

interface ConfidenceBadgeProps {
  probability: number
  showPercentage?: boolean
  className?: string
}

export function ConfidenceBadge({ probability, showPercentage = true, className }: ConfidenceBadgeProps) {
  const info = getConfidenceInfo(probability)

  return (
    <Badge variant="outline" className={cn("font-medium tabular-nums", info.color, info.bgColor, className)}>
      {showPercentage ? formatProbability(probability) : info.label}
    </Badge>
  )
}
