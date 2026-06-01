import { Fragment, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Bot, Send, User } from "lucide-react";
import { toast } from "sonner";
import { API_BASE } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatbotModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const renderInlineFormatting = (text: string) => {
  const segments = text.split(/(\*\*.*?\*\*)/g);
  return segments.map((segment, index) => {
    if (segment.startsWith("**") && segment.endsWith("**")) {
      return (
        <strong key={`${segment}-${index}`} className="font-semibold text-foreground">
          {segment.slice(2, -2)}
        </strong>
      );
    }

    return <Fragment key={`${segment}-${index}`}>{segment}</Fragment>;
  });
};

const renderAssistantContent = (content: string) => {
  const lines = content.split("\n");
  const nodes: JSX.Element[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index].trim();

    if (!line) {
      index += 1;
      continue;
    }

    if (line.startsWith("## ")) {
      nodes.push(
        <h3 key={`heading-${index}`} className="text-sm font-semibold tracking-wide text-primary">
          {line.slice(3)}
        </h3>,
      );
      index += 1;
      continue;
    }

    if (line.startsWith("### ")) {
      nodes.push(
        <h4 key={`subheading-${index}`} className="text-sm font-semibold text-foreground">
          {line.slice(4)}
        </h4>,
      );
      index += 1;
      continue;
    }

    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ""));
        index += 1;
      }

      nodes.push(
        <ol key={`ordered-${index}`} className="list-decimal space-y-1 pl-5 text-sm leading-7">
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineFormatting(item)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ""));
        index += 1;
      }

      nodes.push(
        <ul key={`unordered-${index}`} className="list-disc space-y-1 pl-5 text-sm leading-7">
          {items.map((item, itemIndex) => (
            <li key={`${item}-${itemIndex}`}>{renderInlineFormatting(item)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    const isCautionLine =
      line.toLowerCase().startsWith("medical caution") ||
      line.toLowerCase().startsWith("important") ||
      line.toLowerCase().startsWith("disclaimer");

    if (isCautionLine) {
      nodes.push(
        <div
          key={`caution-${index}`}
          className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-7 text-amber-950"
        >
          {renderInlineFormatting(line)}
        </div>,
      );
      index += 1;
      continue;
    }

    nodes.push(
      <p key={`paragraph-${index}`} className="text-sm leading-7 text-foreground/95">
        {renderInlineFormatting(line)}
      </p>,
    );
    index += 1;
  }

  return <div className="space-y-3">{nodes}</div>;
};

export const ChatbotModal = ({ open, onOpenChange }: ChatbotModalProps) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hello! I'm your medical assistant. I can explain predictions, discuss reports, and answer general medical questions. How can I help?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [pdfFile, setPdfFile] = useState<File | null>(null);

  const appendAssistantChunk = (chunk: string) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;

      const next = [...prev];
      const last = next[next.length - 1];
      if (!last || last.role !== "assistant") {
        return [...next, { role: "assistant", content: chunk }];
      }

      next[next.length - 1] = {
        ...last,
        content: `${last.content}${chunk}`,
      };
      return next;
    });
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    const recentHistory = messages.slice(-8);
    setInput("");
    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage },
      { role: "assistant", content: "" },
    ]);

    try {
      const formData = new FormData();
      formData.append("message", userMessage);
      formData.append("history", JSON.stringify(recentHistory));
      formData.append("stream", "1");
      if (pdfFile) {
        formData.append("pdf", pdfFile);
      }

      const res = await fetch(`${API_BASE}/llm/chat`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || "LLM request failed");
      }

      if (!res.body) {
        const data = await res.json();
        const reply =
          typeof data?.reply === "string"
            ? data.reply
            : "Sorry, I could not generate a response.";
        appendAssistantChunk(reply);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.trim()) continue;

          let payload: any = null;
          try {
            payload = JSON.parse(line);
          } catch {
            continue;
          }

          if (typeof payload?.delta === "string" && payload.delta) {
            appendAssistantChunk(payload.delta);
          }
        }
      }

      if (buffer.trim()) {
        try {
          const payload = JSON.parse(buffer);
          if (typeof payload?.delta === "string" && payload.delta) {
            appendAssistantChunk(payload.delta);
          }
        } catch {
          // Ignore incomplete trailing buffer.
        }
      }
    } catch (error: any) {
      setMessages((prev) => {
        const next = [...prev];
        if (next[next.length - 1]?.role === "assistant" && !next[next.length - 1].content) {
          next.pop();
        }
        return next;
      });
      toast.error(error?.message || "Failed to get response");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl h-[600px] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-primary" />
            Medical Assistant
          </DialogTitle>
          <DialogDescription>
            Ask about medical topics or upload a PDF report for explanation
          </DialogDescription>
        </DialogHeader>

        <ScrollArea className="flex-1 pr-4">
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex gap-3 ${
                  message.role === "assistant" ? "justify-start" : "justify-end"
                }`}
              >
                {message.role === "assistant" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === "assistant"
                      ? "bg-muted/80 border border-border/60 shadow-sm"
                      : "bg-primary text-primary-foreground"
                  }`}
                >
                  {message.role === "assistant" ? (
                    renderAssistantContent(message.content)
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  )}
                </div>
                {message.role === "user" && (
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <User className="h-4 w-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
            {loading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="bg-muted rounded-lg p-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.2s]" />
                    <div className="w-2 h-2 bg-primary rounded-full animate-bounce [animation-delay:0.4s]" />
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="flex gap-2 pt-4 border-t">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={loading}
          />
          <Input
            type="file"
            accept="application/pdf"
            onChange={(e) => setPdfFile(e.target.files?.[0] || null)}
            disabled={loading}
          />
          <Button onClick={handleSend} disabled={loading || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
};
