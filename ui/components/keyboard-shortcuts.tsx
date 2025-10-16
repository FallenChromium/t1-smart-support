"use client"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"

interface KeyboardShortcutsProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function KeyboardShortcuts({ open, onOpenChange }: KeyboardShortcutsProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Горячие клавиши</DialogTitle>
          <DialogDescription>Используйте клавиатуру для быстрой работы с системой</DialogDescription>
        </DialogHeader>
        <div className="space-y-6">
          <div>
            <h3 className="mb-3 text-sm font-semibold">Чат и сообщения</h3>
            <div className="space-y-2">
              <ShortcutItem keys={["Enter"]} description="Отправить сообщение" />
              <ShortcutItem keys={["Shift", "Enter"]} description="Новая строка в сообщении" />
            </div>
          </div>

          <div>
            <h3 className="mb-3 text-sm font-semibold">Работа с рекомендациями</h3>
            <div className="space-y-2">
              <ShortcutItem keys={["Alt", "1"]} description="Выбрать первую рекомендацию" />
              <ShortcutItem keys={["Alt", "2"]} description="Выбрать вторую рекомендацию" />
              <ShortcutItem keys={["Alt", "3"]} description="Выбрать третью рекомендацию" />
              <ShortcutItem keys={["Ctrl", "Shift", "I"]} description="Вставить выбранную рекомендацию в ответ" />
            </div>
          </div>

          <div>
            <h3 className="mb-3 text-sm font-semibold">Навигация</h3>
            <div className="space-y-2">
              <ShortcutItem keys={["Esc"]} description="Снять выделение с рекомендации" />
              <ShortcutItem keys={["Ctrl", "/"]} description="Показать эту справку" />
            </div>
          </div>

          <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground">
            💡 <strong>Совет:</strong> Все шорткаты безопасны и не конфликтуют с системными командами браузера или ОС
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function ShortcutItem({ keys, description }: { keys: string[]; description: string }) {
  return (
    <div className="flex items-center justify-between rounded-lg border p-3">
      <span className="text-sm text-muted-foreground">{description}</span>
      <KbdGroup>
        {keys.map((key, i) => (
          <>
            {i > 0 && <span className="text-muted-foreground">+</span>}
            <Kbd key={key}>{key}</Kbd>
          </>
        ))}
      </KbdGroup>
    </div>
  )
}
