import { FormEvent, useEffect, useRef, useState } from "react";

type IconName = "search" | "compass" | "lightbulb" | "globe" | "link";

type SearchEngine =
  | "auto"
  | "google"
  | "duckduckgo"
  | "bing"
  | "startpage"
  | "brave";
type ModelId =
  | "best"
  | "sonar"
  | "claude-sonnet-4"
  | "claude-sonnet-4-thinking"
  | "claude-opus-4-1-thinking"
  | "gemini-2-5-pro"
  | "gpt-5"
  | "gpt-5-thinking"
  | "o3";

const iconPaths: Record<IconName, string> = {
  search:
    "M12 5.5a1 1 0 0 1 .71.29l4.5 4.5-1.42 1.42L13 9.42V18h-2V9.42l-2.79 2.29-1.42-1.42 4.5-4.5A1 1 0 0 1 12 5.5Z",
  compass:
    "M12 4.5a7.5 7.5 0 1 0 7.5 7.5A7.51 7.51 0 0 0 12 4.5Zm3 4.2-1.21 3.67-3.58 1.47 1.21-3.67Zm-3 8.3a6 6 0 1 1 6-6 6 6 0 0 1-6 6Z",
  lightbulb:
    "M12 3.5A5.5 5.5 0 0 0 8.71 13a2.75 2.75 0 0 0-.71 1.85v.4h8v-.4a2.75 2.75 0 0 0-.71-1.85A5.5 5.5 0 0 0 12 3.5Zm-2.5 13h5a1.5 1.5 0 0 1-1.35 1H10.85A1.5 1.5 0 0 1 9.5 16.5Zm.75 2h3.5a1.25 1.25 0 0 1-1.25 1h-1a1.25 1.25 0 0 1-1.25-1Z",
  globe:
    "M12 4a8 8 0 1 0 8 8 8 8 0 0 0-8-8Zm5.73 7h-2.46a12.43 12.43 0 0 0-1-4.39A6.05 6.05 0 0 1 17.73 11ZM12 6c.62 0 1.73 1.72 2.25 4.5h-4.5C9.27 7.72 11.38 6 12 6ZM6.27 13h2.46a12.43 12.43 0 0 0 1 4.39A6.05 6.05 0 0 1 6.27 13Zm2.46-2H6.27a6.05 6.05 0 0 1 3.46-4.39A12.43 12.43 0 0 0 8.73 11Zm4.27 6c-.62 0-1.73-1.72-2.25-4.5h4.5C14.73 15.28 12.62 17 12 17Zm2.54-.11A12.43 12.43 0 0 0 14.27 13h2.46a6.05 6.05 0 0 1-3.46 4.39Z",
  link: "M9.17 15.12 7.05 17a3 3 0 1 1 0-4.24l1-1 1.06 1.06-1 1a1.5 1.5 0 1 0 2.12 2.12l2.12-1.88 1.01 1.06-2.19 1.92a3 3 0 1 1-2.97-1.92Zm5.66-6.24 2.12-1.88a3 3 0 1 1 0 4.24l-1 1-1.06-1.06 1-1a1.5 1.5 0 1 0-2.12-2.12l-2.12 1.88-1.01-1.06 2.19-1.92a3 3 0 1 1 2.97 1.92Z",
};

const engineConfig: Record<
  SearchEngine,
  { icon: IconName; label: string; description: string }
> = {
  auto: {
    icon: "globe",
    label: "Auto",
    description:
      "Automatically selects the most suitable search engine for your query",
  },
  google: {
    icon: "globe",
    label: "Google Search",
    description: "",
  },
  duckduckgo: {
    icon: "globe",
    label: "DuckDuckGo",
    description: "",
  },
  startpage: {
    icon: "globe",
    label: "Startpage",
    description: "",
  },
  brave: {
    icon: "globe",
    label: "Brave",
    description: "",
  },
  bing: {
    icon: "globe",
    label: "Bing Search",
    description: "",
  },
};

type ModelOption = {
  id: ModelId;
  label: string;
  description?: string;
  badge?: string;
};

const modelOptions: ModelOption[] = [
  {
    id: "best",
    label: "Best",
    description: "Automatically selects the most suitable model for your query",
  },
  {
    id: "sonar",
    label: "Sonar",
    description: "Prioritizes real-time retrieval and fact-checking",
  },
  {
    id: "claude-sonnet-4",
    label: "Claude Sonnet 4.0",
    description: "Balanced reasoning and creativity",
  },
  {
    id: "claude-sonnet-4-thinking",
    label: "Claude Sonnet 4.0 Thinking",
    description: "Longer reasoning chains and transparent thinking process",
  },
  {
    id: "claude-opus-4-1-thinking",
    label: "Claude Opus 4.1 Thinking",
    description: "Deep reasoning capability",
    badge: "Max",
  },
  {
    id: "gemini-2-5-pro",
    label: "Gemini 2.5 Pro",
    description: "Google flagship model",
  },
  {
    id: "gpt-5",
    label: "GPT-5",
    description: "Suited for general analysis and generation",
  },
  {
    id: "gpt-5-thinking",
    label: "GPT-5 Thinking",
    description: "Structured reasoning with step-by-step outputs",
  },
  {
    id: "o3",
    label: "o3",
    description: "OpenAI advanced reasoning model",
  },
];

const SearchBox = () => {
  const [query, setQuery] = useState("");
  const [engine, setEngine] = useState<SearchEngine>("duckduckgo");
  const [engineMenuOpen, setEngineMenuOpen] = useState(false);
  const [model, setModel] = useState<ModelId>("best");
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const [output, setOutput] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [events, setEvents] = useState<Array<{ type: string; payload: any }>>(
    [],
  );
  const [errorMessage, setErrorMessage] = useState("");
  const streamRef = useRef<HTMLDivElement | null>(null);
  const STREAM_MAX_EVENTS = 200;
  const [status, setStatus] = useState<{
    stage?: string;
    message?: string;
    elapsed?: number;
  } | null>(null);
  const [clarPrompt, setClarPrompt] = useState<string | null>(null);
  const [clarText, setClarText] = useState<string>("");
  const [clarRewritten, setClarRewritten] = useState<{
    original?: string;
    rewritten?: string;
    added?: string;
  } | null>(null);
  const canSubmit = query.trim().length > 0;
  const isRunning = Boolean(sessionId) && Boolean(status);
  // Track cancelled session ids to ignore late responses
  const cancelledSidRef = useRef<Set<string>>(new Set());
  // Whether UI should show active run info (status/streams)
  const [idle, setIdle] = useState(true);

  const engineButtonRef = useRef<HTMLButtonElement | null>(null);
  const engineMenuRef = useRef<HTMLDivElement | null>(null);
  const modelButtonRef = useRef<HTMLButtonElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!engineMenuOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        engineMenuRef.current &&
        !engineMenuRef.current.contains(event.target as Node) &&
        engineButtonRef.current &&
        !engineButtonRef.current.contains(event.target as Node)
      ) {
        setEngineMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [engineMenuOpen]);

  useEffect(() => {
    if (!modelMenuOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        !modelMenuRef.current.contains(event.target as Node) &&
        modelButtonRef.current &&
        !modelButtonRef.current.contains(event.target as Node)
      ) {
        setModelMenuOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [modelMenuOpen]);

  // Listen for streamed reasoning events
  useEffect(() => {
    const handler = (_: any, evt: any) => {
      if (!evt) return;
      // Filter by active session
      if (sessionId && evt.sessionId === sessionId) {
        if (evt.type === "status") {
          setStatus({
            stage: evt.payload?.stage,
            message: evt.payload?.message,
            elapsed: evt.payload?.elapsed,
          });
        } else if (evt.type === "started") {
          setStatus({ stage: "started", message: "Starting…" });
        } else if (evt.type === "reasoning") {
          // Keep status aligned with streaming reasoning
          setStatus((prev) =>
            prev?.stage === "clarification" ? prev : { stage: "reasoning", message: "Reasoning…" },
          );
        } else if (evt.type === "search_result") {
          // Reflect receipt of search results
          setStatus((prev) =>
            prev?.stage === "clarification" ? prev : { stage: "search", message: "Analyzing search results…" },
          );
        } else if (evt.type === "outline_update" || evt.type === "write_section_plan") {
          // Indicate writing phase progress
          setStatus((prev) =>
            prev?.stage === "clarification" ? prev : { stage: "writing", message: "Writing report…" },
          );
        } else if (evt.type === "final") {
          setStatus(null);
          setSessionId(null);
          setClarPrompt(null);
          setClarText("");
          setClarRewritten(null);
        } else if (evt.type === "cancelled") {
          setEvents((prev) => [
            ...prev,
            { type: "cancelled", payload: evt.payload },
          ]);
          setStatus(null);
          setSessionId(null);
          setClarPrompt(null);
          setClarText("");
          setClarRewritten(null);
        } else if (evt.type === "clarification_request") {
          const msg =
            typeof evt.payload === "object"
              ? evt.payload?.message
              : String(evt.payload ?? "");
          const text =
            msg || "Please provide additional details to clarify your request.";
          setClarPrompt(text);
          // During clarification, mark an awaiting status so users understand why progression pauses
          setStatus({ stage: "clarification", message: "Awaiting clarification…" });
          setEvents((prev) => [
            ...prev,
            { type: "clarification_request", payload: text },
          ]);
        } else if (evt.type === "clarification_rewrite") {
          const payload = evt.payload || {};
          const rewritten = payload.rewritten;
          setClarRewritten({
            original: payload.original,
            rewritten,
            added: payload.added,
          });
          if (rewritten) {
            setEvents((prev) => [
              ...prev,
              { type: "clarified_query", payload: rewritten },
            ]);
          }
          // Resume reasoning after clarification has been processed
          setStatus({ stage: "reasoning", message: "Reasoning…" });
        } else if (evt.type) {
          setEvents((prev) => [
            ...prev,
            { type: evt.type, payload: evt.payload },
          ]);
        }
      }
    };
    window.ipcRenderer.on("reasoning-event", handler);
    return () => {
      window.ipcRenderer.off("reasoning-event", handler);
    };
  }, [sessionId]);

  // Auto-scroll stream box to bottom when new events arrive
  useEffect(() => {
    if (streamRef.current) {
      streamRef.current.scrollTop = streamRef.current.scrollHeight;
    }
  }, [events]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    // If backend requested clarification, reuse the bottom input to send it
    if (clarPrompt && sessionId) {
      const text = query.trim();
      if (!text) return;
      try {
        // Clear input immediately so the clarification text doesn't linger
        setQuery("");
        await window.ipcRenderer.invoke("send-clarification", {
          sessionId,
          text,
        });
        setEvents((prev) => [
          ...prev,
          { type: "clarification_sent", payload: text },
        ]);
        setClarPrompt(null);
        setClarText("");
      } catch (e) {
        setEvents((prev) => [
          ...prev,
          {
            type: "clarification-error",
            payload:
              e instanceof Error ? e.message : "Failed to send clarification",
          },
        ]);
        // Still clear local input on error to avoid stale text
        setQuery("");
      }
      return;
    }

    const trimmed = query.trim();
    if (!trimmed) return;

    try {
      setErrorMessage("");
      setOutput("");
      setEvents([]);
      setIdle(false);
      setStatus({ stage: "started", message: "Starting…" });
      const sid =
        (globalThis.crypto as any)?.randomUUID?.() || String(Date.now());
      setSessionId(sid);
      // Clear the input immediately on submit so the text
      // does not linger in the search box while the request runs
      setQuery("");
      const response: { sessionId: string; result?: any } | undefined =
        await window.ipcRenderer.invoke("submit-query", {
          engine,
          model,
          query: trimmed,
          sessionId: sid,
        });
      // If this session was cancelled, ignore any late response
      if (cancelledSidRef.current.has(sid)) {
        cancelledSidRef.current.delete(sid);
        return;
      }
      setEngineMenuOpen(false);
      setModelMenuOpen(false);
      if (response?.sessionId === sid) {
        if (response?.result) {
          setOutput(JSON.stringify(response.result, null, 2));
        }
      } else if (response) {
        setOutput(JSON.stringify(response, null, 2));
      }
    } catch (error) {
      console.error("Failed to submit request", error);
      setErrorMessage(
        error instanceof Error ? error.message : "Failed to submit request",
      );
      // Ensure the input is cleared even if the request fails
      setQuery("");
    }
  };

  const handleCancel = async () => {
    if (!sessionId) return;
    try {
      // mark immediately as cancelled in UI to allow new input
      cancelledSidRef.current.add(sessionId);
      // reset UI to initial state
      setStatus(null);
      setEvents([]);
      setOutput("");
      setErrorMessage("");
      setSessionId(null);
      setIdle(true);
      setQuery("");
      setEngineMenuOpen(false);
      setModelMenuOpen(false);
      await window.ipcRenderer.invoke("cancel-session", { sessionId });
    } catch (e) {
      // Surface cancel failure softly
      setEvents((prev) => [
        ...prev,
        {
          type: "cancel-error",
          payload: e instanceof Error ? e.message : "Cancel failed",
        },
      ]);
    }
  };

  const submitClarification = async () => {
    if (!sessionId) return;
    const text = (clarText || "").trim();
    if (!text) return;
    try {
      await window.ipcRenderer.invoke("send-clarification", {
        sessionId,
        text,
      });
      // lock UI until backend proceeds; clear local inputs
      setClarPrompt(null);
      setClarText("");
    } catch (e) {
      setEvents((prev) => [
        ...prev,
        {
          type: "clarification-error",
          payload:
            e instanceof Error ? e.message : "Failed to send clarification",
        },
      ]);
    }
  };

  const formRef = useRef<HTMLFormElement | null>(null);

  return (
    <section
      className="search-wrapper"
      aria-label="Deep Research prompt"
      style={{ paddingBottom: !idle ? 120 : 0 }}
    >
      <form
        ref={formRef}
        className={`search-shell${!idle ? " fixed-bottom" : ""}`}
        onSubmit={handleSubmit}
      >
        <input
          className="search-input"
          placeholder={
            clarPrompt ? "Please provide more details…" : "Ask anything..."
          }
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label={clarPrompt ? "Provide clarification" : "Ask a question"}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              const f = formRef.current as any;
              if (f && typeof f.requestSubmit === "function") {
                f.requestSubmit();
              } else {
                const btn = document.createElement("button");
                btn.type = "submit";
                btn.style.display = "none";
                formRef.current?.appendChild(btn);
                btn.click();
                formRef.current?.removeChild(btn);
              }
            }
          }}
        />
        <div className="search-inline-actions">
          <div className="menu-selector engine-selector" ref={engineMenuRef}>
            <button
              ref={engineButtonRef}
              className={`inline-icon-btn engine-btn${engineMenuOpen ? " active" : ""}`}
              type="button"
              aria-label={`Select search engine (current: ${engineConfig[engine].label})`}
              onClick={() => {
                setModelMenuOpen(false);
                setEngineMenuOpen((prev) => !prev);
              }}
              title={`当前：${engineConfig[engine].label}`}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d={iconPaths[engineConfig[engine].icon]} />
              </svg>
            </button>
            {engineMenuOpen ? (
              <div className="menu-dropdown engine-menu" role="menu">
                {(Object.keys(engineConfig) as Array<SearchEngine>).map(
                  (option) => (
                    <button
                      key={option}
                      className={`menu-item${engine === option ? " selected" : ""}`}
                      type="button"
                      role="menuitemradio"
                      aria-checked={engine === option}
                      onClick={() => {
                        setEngine(option);
                        setEngineMenuOpen(false);
                      }}
                    >
                      <div className="menu-item-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24">
                          <path d={iconPaths[engineConfig[option].icon]} />
                        </svg>
                      </div>
                      <div className="menu-item-labels">
                        <span className="menu-item-title">
                          {engineConfig[option].label}
                        </span>
                        <small className="menu-item-desc">
                          {engineConfig[option].description}
                        </small>
                      </div>
                      <span
                        className={`menu-toggle${engine === option ? " on" : ""}`}
                        aria-hidden="true"
                      />
                    </button>
                  ),
                )}
              </div>
            ) : null}
          </div>
          <div className="menu-selector model-selector" ref={modelMenuRef}>
            <button
              ref={modelButtonRef}
              className={`inline-icon-btn model-btn${modelMenuOpen ? " active" : ""}`}
              type="button"
              aria-label="选择模型"
              onClick={() => {
                setEngineMenuOpen(false);
                setModelMenuOpen((prev) => !prev);
              }}
              title={modelOptions.find((option) => option.id === model)?.label}
            >
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d={iconPaths.compass} />
              </svg>
            </button>
            {modelMenuOpen ? (
              <div className="menu-dropdown model-menu" role="menu">
                {modelOptions.map((option) => (
                  <button
                    key={option.id}
                    className={`menu-item${model === option.id ? " selected" : ""}`}
                    type="button"
                    role="menuitemradio"
                    aria-checked={model === option.id}
                    onClick={() => {
                      setModel(option.id);
                      setModelMenuOpen(false);
                    }}
                  >
                    <div className="menu-item-labels">
                      <span className="menu-item-title">
                        {option.label}
                        {option.badge ? (
                          <span className="menu-item-badge" aria-hidden="true">
                            {option.badge}
                          </span>
                        ) : null}
                      </span>
                      {option.description ? (
                        <small className="menu-item-desc">
                          {option.description}
                        </small>
                      ) : null}
                    </div>
                    <span
                      className={`menu-toggle${model === option.id ? " on" : ""}`}
                      aria-hidden="true"
                    />
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
        {!idle && status?.message ? (
          <span
            className="inline-status"
            title={status.stage || ""}
            aria-live="polite"
          >
            <span className="spinner" aria-hidden="true" />
            <span className="status-text">
              {status.message}
              {typeof status.elapsed === "number"
                ? ` (${status.elapsed}s)`
                : ""}
            </span>
          </span>
        ) : null}
        {clarPrompt && sessionId ? (
          <button
            className="search-icon-btn"
            type="submit"
            aria-label="Submit clarification"
            disabled={!canSubmit}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d={iconPaths.search} />
            </svg>
          </button>
        ) : isRunning ? (
          <button
            className="search-icon-btn stop"
            type="button"
            aria-label="Stop"
            onClick={handleCancel}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              {/* stop square */}
              <path d="M7 7h10v10H7z" />
            </svg>
          </button>
        ) : (
          <button
            className="search-icon-btn"
            type="submit"
            aria-label="Search"
            disabled={!canSubmit}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d={iconPaths.search} />
            </svg>
          </button>
        )}
      </form>
      <p className="search-helper" style={{ display: idle ? "block" : "none" }}>
        Discover facts, synthesize insights, and craft in-depth plans.
      </p>
      {!idle && events.length > 0 ? (
        <div
          className="search-output stream-output"
          role="status"
          aria-live="polite"
          ref={streamRef}
        >
          <span className="search-output-label">Reasoning Stream</span>
          <pre>
            {events
              .slice(-STREAM_MAX_EVENTS)
              .map(
                (e) =>
                  `${e.type}: ${typeof e.payload === "string" ? e.payload : JSON.stringify(e.payload)}`,
              )
              .join("\n")}
          </pre>
        </div>
      ) : null}
      {/* Clarification requests are shown in stream; bottom input is reused to answer */}
      {/* Clarified Query is now streamed into Reasoning Stream as 'clarified_query' event */}
      {!idle && output ? (
        <div className="search-output" role="status" aria-live="polite">
          <span className="search-output-label">Report</span>
          <pre>{output}</pre>
        </div>
      ) : null}
      {!idle && errorMessage ? (
        <div className="search-output error" role="alert">
          <span className="search-output-label">Error</span>
          <pre>{errorMessage}</pre>
        </div>
      ) : null}
    </section>
  );
};

export default SearchBox;
