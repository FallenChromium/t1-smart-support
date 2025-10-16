import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ConfidenceBadge } from "./confidence-badge"

interface CategoryDisplayProps {
  category: string
  probabilities: Record<string, number>
}

export function CategoryDisplay({ category, probabilities }: CategoryDisplayProps) {
  const topCategories = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 3)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Классификация обращения</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Категория:</span>
          <Badge variant="secondary" className="font-semibold">
            {category}
          </Badge>
        </div>
        <div className="space-y-2">
          <span className="text-sm font-medium text-muted-foreground">Распределение вероятностей:</span>
          {topCategories.map(([cat, prob]) => (
            <div key={cat} className="flex items-center justify-between gap-4">
              <span className="text-sm text-muted-foreground">{cat}</span>
              <ConfidenceBadge probability={prob} />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
