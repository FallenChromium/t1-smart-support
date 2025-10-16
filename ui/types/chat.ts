export interface Message {
  id: string
  role: "customer" | "agent"
  content: string
  timestamp: Date
}

export interface Recommendation {
  id: string
  query: string
  answer: string
  similarity: number
  category?: string
}
