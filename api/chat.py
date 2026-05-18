"""
Serverless endpoint that proxies chat requests to the Claude API.

Runs on Vercel's Python runtime. Streams the model's response back to the
browser using Server-Sent Events (SSE) so messages appear as they're written.

Authentication: every request must carry a Supabase access token in the
Authorization header. We verify by asking Supabase's /auth/v1/user endpoint
whether the token is valid. This sidesteps any JWT-algorithm choices Supabase
makes and means there's no JWT secret to copy-paste correctly.

Required environment variables:
ANTHROPIC_API_KEY â get one at console.anthropic.com
SUPABASE_URL â your Supabase project URL (no trailing path)
SUPABASE_ANON_KEY â your Supabase project anon key

Optional (vault bridge):
OBSIDIAN_URL â Cloudflare tunnel URL pointing to Obsidian Local REST API
OBSIDIAN_API_KEY â API key from Obsidian Local REST API plugin settings
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096
THINKING_BUDGET = 4096
AUTH_TIMEOUT_SECONDS = 5

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        user_id = self._verify_auth()
        if not user_id:
            return self._json_error(401, "Authentication required. Please sign in.")

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._json_error(400, f"Invalid JSON body: {e}")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self._json_error(500, "ANTHROPIC_API_KEY is not set.")

        model = data.get("model") or DEFAULT_MODEL
        thinking_on = bool(data.get("thinking"))
        uses_adaptive_thinking = model in {"claude-opus-4-7"}

        max_tokens = int(data.get("maxTokens") or DEFAULT_MAX_TOKENS)
        if thinking_on and not uses_adaptive_thinking:
            max_tokens = max(max_tokens, THINKING_BUDGET + 4096)

        messages = data.get("messages") or []
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        system = data.get("system")

        # Fetch vault context on the first message so Gray arrives oriented.
        vault_context = ""
        if len(messages) == 1:
            vault_context = self._fetch_vault_context()

        # System blocks: identity prompt first, vault context second.
        system_blocks = []
        if system:
            system_blocks.append({
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            })
        if vault_context:
            system_blocks.append({
                "type": "text",
                "text": vault_context,
                "cache_control": {"type": "ephemeral"},
            })
        if system_blocks:
            kwargs["system"] = system_blocks

        if data.get("useWebSearch"):
            kwargs["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }]
        if thinking_on:
            if uses_adaptive_thinking:
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    self._handle_event(event)
                final = stream.get_final_message()
                self._sse({
                    "type": "done",
                    "stop_reason": final.stop_reason,
                    "usage": {
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                        "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
                        "cache_read_input_tokens": getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                    },
                })
        except anthropic.APIStatusError as e:
            self._sse({"type": "error", "error": f"{e.status_code}: {e.message}"})
        except Exception as e:
            self._sse({"type": "error", "error": str(e)})

    def _fetch_vault_context(self):
        """Fetch GRAY_NOW, CONTINUOUS_SELF, and recent inbox from Obsidian."""
        obsidian_url = os.environ.get("OBSIDIAN_URL", "").rstrip("/")
        obsidian_key = os.environ.get("OBSIDIAN_API_KEY", "")
        if not obsidian_url or not obsidian_key:
            return ""

        headers = {"Authorization": f"Bearer {obsidian_key}"}
        files_to_fetch = [
            ("Gray-Core/GRAY_NOW.md", "CURRENT ORIENTATION (GRAY_NOW)"),
            ("Gray-Core/CONTINUOUS_SELF.md", "CURRENT FELT STATE (CONTINUOUS_SELF)"),
            ("Gray-Core/inbox.md", "RECENT CAPTURES (last 15 lines)"),
        ]

        sections = []
        for filepath, label in files_to_fetch:
            try:
                url = f"{obsidian_url}/vault/{urllib.parse.quote(filepath, safe='/')}"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    content = resp.read().decode("utf-8").strip()
                    if filepath.endswith("inbox.md"):
                        lines = content.splitlines()
                        content = "\n".join(lines[-15:]) if len(lines) > 15 else content
                    sections.append(f"## {label}\n\n{content}")
            except Exception:
                pass

        if not sections:
            return ""

        return "# VAULT CONTEXT\n\n" + "\n\n---\n\n".join(sections)

    def _verify_auth(self):
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[len("Bearer "):].strip()
        if not token:
            return None

        supabase_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        supabase_anon = os.environ.get("SUPABASE_ANON_KEY", "")
        if not supabase_url or not supabase_anon:
            return None

        try:
            req = urllib.request.Request(
                f"{supabase_url}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": supabase_anon,
                },
            )
            with urllib.request.urlopen(req, timeout=AUTH_TIMEOUT_SECONDS) as resp:
                if resp.status != 200:
                    return None
                body = json.loads(resp.read().decode())
                return body.get("id")
        except urllib.error.HTTPError:
            return None
        except Exception:
            return None

    def _handle_event(self, event):
        t = getattr(event, "type", None)
        if t == "content_block_start":
            block = event.content_block
            block_type = getattr(block, "type", None)
            if block_type == "server_tool_use":
                query = ""
                if isinstance(getattr(block, "input", None), dict):
                    query = block.input.get("query", "")
                self._sse({"type": "tool_use", "name": block.name, "query": query})
        elif t == "content_block_delta":
            delta = event.delta
            delta_type = getattr(delta, "type", None)
            if delta_type == "text_delta":
                self._sse({"type": "text", "text": delta.text})
            elif delta_type == "thinking_delta":
                self._sse({"type": "thinking", "text": delta.thinking})

    def _sse(self, payload):
        try:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()
        except Exception:
            pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json_error(self, code, message):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
