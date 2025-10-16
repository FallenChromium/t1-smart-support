// Utility functions for confidence levels and colors

export type ConfidenceLevel = "high" | "medium" | "low"

export interface ConfidenceInfo {
  level: ConfidenceLevel
  color: string
  bgColor: string
  label: string
  description: string
}

export function getConfidenceLevel(probability: number): ConfidenceLevel {
  if (probability >= 0.7) return "high"
  if (probability >= 0.4) return "medium"
  return "low"
}

export function getConfidenceInfo(probability: number): ConfidenceInfo {
  const level = getConfidenceLevel(probability)

  const configs: Record<ConfidenceLevel, ConfidenceInfo> = {
    high: {
      level: "high",
      color: "text-emerald-700 dark:text-emerald-400",
      bgColor: "bg-emerald-100 dark:bg-emerald-950 border-emerald-300 dark:border-emerald-800",
      label: "Высокая уверенность",
      description: "Модель уверена в этой рекомендации",
    },
    medium: {
      level: "medium",
      color: "text-amber-700 dark:text-amber-400",
      bgColor: "bg-amber-100 dark:bg-amber-950 border-amber-300 dark:border-amber-800",
      label: "Средняя уверенность",
      description: "Рекомендация требует проверки",
    },
    low: {
      level: "low",
      color: "text-rose-700 dark:text-rose-400",
      bgColor: "bg-rose-100 dark:bg-rose-950 border-rose-300 dark:border-rose-800",
      label: "Низкая уверенность",
      description: "Требуется ручная обработка",
    },
  }

  return configs[level]
}

export function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`
}
