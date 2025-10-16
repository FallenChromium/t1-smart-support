"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { SendIcon, UserIcon } from "lucide-react"
import { Input } from "@/components/ui/input"

interface ChatInputProps {
  onSend: (message: string) => void
  onSimulateCustomer: (message: string) => void
  disabled?: boolean
  placeholder?: string
  value?: string
  onValueChange?: (value: string) => void
}

export function ChatInput({
  onSend,
  onSimulateCustomer,
  disabled,
  placeholder,
  value: externalValue,
  onValueChange,
}: ChatInputProps) {
  const [internalValue, setInternalValue] = useState("")
  const [customerMessage, setCustomerMessage] = useState("")
  const message = externalValue !== undefined ? externalValue : internalValue
  const setMessage = onValueChange || setInternalValue

  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSend = () => {
    if (message.trim() && !disabled) {
      onSend(message)
      if (!onValueChange) {
        setInternalValue("")
      }
    }
  }

  const handleSimulateCustomer = () => {
    if (customerMessage.trim() && !disabled) {
      onSimulateCustomer(customerMessage)
      setCustomerMessage("")
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleCustomerKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault()
      handleSimulateCustomer()
    }
  }

  return (
    <div className="border-t bg-background p-4">
      <div className="mx-auto max-w-4xl space-y-3">
        <div className="flex gap-2">
          <Input
            value={customerMessage}
            onChange={(e) => setCustomerMessage(e.target.value)}
            onKeyDown={handleCustomerKeyDown}
            placeholder="Симулировать сообщение от клиента..."
            disabled={disabled}
            className="flex-1"
          />
          <Button variant="secondary" onClick={handleSimulateCustomer} disabled={!customerMessage.trim() || disabled}>
            <UserIcon className="h-4 w-4" />
            Отправить от клиента
          </Button>
        </div>

        <div className="relative">
          <Textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder || "Введите ответ клиенту..."}
            disabled={disabled}
            rows={3}
            className="resize-none pr-12"
          />
          <Button
            size="icon"
            onClick={handleSend}
            disabled={!message.trim() || disabled}
            className="absolute bottom-2 right-2"
          >
            <SendIcon className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="flex items-center gap-1">
              <Kbd>Enter</Kbd> отправить
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> новая строка
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Ctrl</Kbd>+<Kbd>Shift</Kbd>+<Kbd>I</Kbd> вставить рекомендацию
            </span>
            <span className="flex items-center gap-1">
              <Kbd>Ctrl</Kbd>+<Kbd>/</Kbd> все шорткаты
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
