import { cn } from "@/lib/utils"
import { UserIcon, HeadsetIcon } from "lucide-react"
import type { Message } from "@/types/chat"

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isCustomer = message.role === "customer"

  return (
    <div className={cn("flex gap-3 p-4", isCustomer ? "bg-muted/30" : "bg-background")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isCustomer
            ? "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300"
            : "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
        )}
      >
        {isCustomer ? <UserIcon className="h-4 w-4" /> : <HeadsetIcon className="h-4 w-4" />}
      </div>
      <div className="flex-1 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">{isCustomer ? "Клиент" : "Вы (Оператор)"}</span>
          <span className="text-xs text-muted-foreground">
            {message.timestamp.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" })}
          </span>
        </div>
        <p className="text-pretty text-sm leading-relaxed">{message.content}</p>
      </div>
    </div>
  )
}
