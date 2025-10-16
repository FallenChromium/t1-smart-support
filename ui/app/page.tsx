"use client"

import { useState, useEffect, useRef } from "react"
import { ChatMessage } from "@/components/chat-message"
import { ChatInput } from "@/components/chat-input"
import { RecommendationsPanel } from "@/components/recommendations-panel"
import { KeyboardShortcuts } from "@/components/keyboard-shortcuts"
import { predictCategory, findTopAnswers } from "@/lib/api-client"
import type { Message, Recommendation } from "@/types/chat"
import { useToast } from "@/hooks/use-toast"
import { Button } from "@/components/ui/button"
import { HelpCircleIcon } from "lucide-react"
import { Kbd } from "@/components/ui/kbd"

const SAMPLE_CUSTOMER_MESSAGES = [
  "Здравствуйте! Я не могу войти в свой аккаунт, пишет что пароль неверный, хотя я точно помню его.",
  "Добрый день! У меня не работает оплата картой, постоянно выдает ошибку. Что делать?",
  "Привет! Хочу вернуть товар, который заказал на прошлой неделе. Как это сделать?",
  "Здравствуйте! Не приходит код подтверждения на телефон уже 20 минут. Помогите!",
]

export default function SupportAssistant() {
  const [messages, setMessages] = useState<Message[]>([])
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [showShortcuts, setShowShortcuts] = useState(false)
  const [category, setCategory] = useState<string>("")
  const [inputValue, setInputValue] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const analyzeMessage = async (content: string) => {
    setIsAnalyzing(true)
    setRecommendations([])
    setSelectedIndex(null)
    setCategory("")

    try {
      const [predictionResult, answersResult] = await Promise.all([predictCategory(content), findTopAnswers(content)])

      const categoryLabelParts = [predictionResult.prediction]
      if (predictionResult.parent && predictionResult.parent !== predictionResult.prediction) {
        categoryLabelParts.push(`parent: ${predictionResult.parent}`)
      }
      if (typeof predictionResult.confidence === "number") {
        const confidencePercent = Math.round(predictionResult.confidence * 100)
        categoryLabelParts.push(`confidence: ${confidencePercent}%`)
      }
      setCategory(categoryLabelParts.filter(Boolean).join(" • "))
      setRecommendations(
        answersResult.answers.map((answer, index) => ({
          id: `rec-${Date.now()}-${index}`,
          query: answer.matchedQuery,
          answer: answer.retrievedAnswer,
          similarity: answer.similarityScore,
        })),
      )
    } catch (error) {
      console.error("[v0] Analysis error:", error)
      toast({
        title: "Ошибка анализа",
        description: "Не удалось получить рекомендации. Проверьте подключение к API.",
        variant: "destructive",
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSimulateCustomer = (content: string) => {
    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role: "customer",
      content,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
    analyzeMessage(content)
  }

  const handleSendMessage = (content: string) => {
    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role: "agent",
      content,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
    setInputValue("")
  }

  const handleInsertRecommendation = (answer: string) => {
    setInputValue((prev) => (prev ? `${prev}\n\n${answer}` : answer))
    toast({
      title: "Рекомендация вставлена",
      description: "Текст добавлен в поле ввода",
    })
  }

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+/ - show shortcuts help
      if (e.ctrlKey && e.key === "/") {
        e.preventDefault()
        setShowShortcuts(true)
      }

      // Alt+1/2/3 - select recommendation
      if (e.altKey && ["1", "2", "3"].includes(e.key) && recommendations.length > 0) {
        const index = Number.parseInt(e.key) - 1
        if (index < recommendations.length) {
          e.preventDefault()
          setSelectedIndex(index)
        }
      }

      // Ctrl+Shift+I - insert selected recommendation
      if (e.ctrlKey && e.shiftKey && e.key === "I" && selectedIndex !== null && recommendations[selectedIndex]) {
        e.preventDefault()
        handleInsertRecommendation(recommendations[selectedIndex].answer)
      }

      // Escape - deselect
      if (e.key === "Escape") {
        setSelectedIndex(null)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [recommendations, selectedIndex])

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <header className="flex items-center justify-between border-b bg-card px-6 py-3 shadow-sm">
        <div>
          <h1 className="text-balance text-xl font-bold">AI Ассистент Поддержки</h1>
          <p className="text-xs text-muted-foreground">Интеллектуальные рекомендации в реальном времени</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => setShowShortcuts(true)}>
          <HelpCircleIcon className="h-4 w-4" />
          Шорткаты
          <Kbd>Ctrl</Kbd>+<Kbd>/</Kbd>
        </Button>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat area */}
        <div className="flex flex-1 flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center p-8">
                <div className="max-w-md text-center">
                  <h2 className="mb-2 text-lg font-semibold">Добро пожаловать!</h2>
                  <p className="mb-4 text-pretty text-sm text-muted-foreground">
                    Введите сообщение от клиента в поле внизу, чтобы увидеть AI-рекомендации в действии
                  </p>
                  <div className="space-y-2 text-xs text-muted-foreground">
                    <p>AI автоматически анализирует каждое сообщение клиента и предлагает релевантные ответы</p>
                    <p>Используйте клавиатурные сокращения для быстрой работы</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-px">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} />
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <ChatInput
            onSend={handleSendMessage}
            onSimulateCustomer={handleSimulateCustomer}
            disabled={isAnalyzing}
            placeholder="Введите ответ клиенту..."
            value={inputValue}
            onValueChange={setInputValue}
          />
        </div>

        {/* Recommendations panel */}
        <div className="w-96">
          <RecommendationsPanel
            recommendations={recommendations}
            isLoading={isAnalyzing}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
            onInsert={handleInsertRecommendation}
            category={category}
          />
        </div>
      </div>

      <KeyboardShortcuts open={showShortcuts} onOpenChange={setShowShortcuts} />
    </div>
  )
}
