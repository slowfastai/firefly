#!/usr/bin/env python3
"""Event-stream handler that runs deepresearch and prints JSONL events."""

import sys
import json
import uuid
import asyncio
import threading
import os
import signal

# Allow importing from the monorepo
from pathlib import Path

# Resolve repo root (apps/desktop/scripts -> desktop -> apps -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[3]

# Ensure backend sources are importable (run_backend.py lives here)
BACKEND_SRC = REPO_ROOT / "apps" / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


def main() -> None:
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else "{}"
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        payload = {"query": raw_payload}

    question = (payload.get("query") or "").strip()
    if not question:
        print(
            json.dumps(
                {"type": "error", "payload": {"message": "Empty query"}},
                ensure_ascii=False,
            )
        )
        return

    session_id = payload.get("session_id") or uuid.uuid4().hex

    # Optional args overrides from renderer
    args_overrides = payload.get("args") or {}

    def emit(event: dict) -> None:
        event = dict(event or {})
        event.setdefault("type", "log")
        event.setdefault("payload", {})
        event["session_id"] = session_id
        print(json.dumps(event, ensure_ascii=False), flush=True)

    # Immediately notify UI this session has started
    emit(
        {
            "type": "started",
            "payload": {
                "question": question,
                "engine": payload.get("engine"),
                "model": payload.get("model"),
                "timestamp": __import__("time").time(),
            },
        }
    )

    # Fast-exit on SIGTERM/SIGINT to cooperate with UI cancel
    def _handle_term(_signum, _frame):
        try:
            print(
                json.dumps(
                    {
                        "type": "cancelled",
                        "payload": {"message": "Cancelled by signal"},
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        finally:
            sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)

        # Also listen on stdin for cooperative cancel and interactive messages from the Electron main process
        from threading import Event

        _clar_event = Event()
        _clar_value = {"payload": None}

        def _stdin_listener():
            try:
                for line in sys.stdin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        if isinstance(msg, dict) and msg.get("type") == "cancel":
                            print(
                                json.dumps(
                                    {
                                        "type": "cancelled",
                                        "payload": {"message": "Cancelled by user"},
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                            os._exit(0)
                        elif (
                            isinstance(msg, dict)
                            and msg.get("type") == "clarification_answer"
                        ):
                            _clar_value["payload"] = msg.get("payload") or msg
                            _clar_event.set()
                            # forward a log for visibility
                            try:
                                print(
                                    json.dumps(
                                        {
                                            "type": "log",
                                            "payload": {
                                                "message": "clarification answer received"
                                            },
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                            except Exception:
                                pass
                    except Exception:
                        # Unstructured line; ignore
                        continue
            except Exception:
                pass

        threading.Thread(
            target=_stdin_listener, name="stdin-cancel-listener", daemon=True
        ).start()

        # Mark streaming mode to disable Python stdout logging and keep logs in file only
        os.environ["DR_STREAMING"] = "1"
        # Lazy import so we can emit a structured error if it fails
        from run_backend import run_sequence  # type: ignore

        async def runner():
            async def await_user_reply(_request: dict):
                while not _clar_event.is_set():
                    await asyncio.sleep(0.1)
                _clar_event.clear()
                return _clar_value.get("payload")

            result = await run_sequence(
                question=question,
                args=args_overrides,
                emit=emit,
                await_user_reply=await_user_reply,
            )
            # Emit final event with the original result shape
            emit({"type": "final", "payload": result})

        asyncio.run(runner())
    except Exception as e:
        emit({"type": "error", "payload": {"message": str(e)}})


if __name__ == "__main__":
    main()
