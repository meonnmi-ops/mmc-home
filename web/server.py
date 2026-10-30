#!/usr/bin/env python3
"""Nyanlin Chat Web Server — Termux-friendly"""
import os, subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler
from kb import match_kb

MODEL = os.path.expanduser("~/models/nyanlin/nyanlin-brain-v1-q4_0.gguf")
CHAT = os.path.expanduser("~/mmc_publish/build/chat_v14")
PORT = 8080

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nyanlin Chat</title>
<style>
* { box-sizing: border-box; }
body {
  font-family: -apple-system, sans-serif;
  background: #0a0e14;
  color: #e6e6e6;
  margin: 0;
  padding: 0;
  height: 100vh;
  display: flex;
  flex-direction: column;
}
header {
  background: #1a1f28;
  padding: 12px 16px;
  border-bottom: 1px solid #2a3038;
  font-weight: bold;
  color: #00ff88;
}
#chat {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.msg {
  margin: 8px 0;
  padding: 10px 14px;
  border-radius: 12px;
  max-width: 80%;
  word-wrap: break-word;
  white-space: pre-wrap;
}
.you {
  background: #1e40af;
  margin-left: auto;
  text-align: right;
}
.nyanlin {
  background: #1a1f28;
  border-left: 3px solid #00ff88;
}
form {
  display: flex;
  padding: 12px;
  background: #1a1f28;
  border-top: 1px solid #2a3038;
}
input {
  flex: 1;
  padding: 12px;
  border-radius: 8px;
  border: 1px solid #2a3038;
  background: #0a0e14;
  color: #e6e6e6;
  font-size: 16px;
}
button {
  padding: 12px 20px;
  margin-left: 8px;
  border-radius: 8px;
  border: none;
  background: #00ff88;
  color: #0a0e14;
  font-weight: bold;
  font-size: 16px;
  cursor: pointer;
}
button:disabled { opacity: 0.5; }
</style>
</head>
<body>
<header>🐉 Nyanlin Chat — 100% Pure MMC</header>
<div id="chat"></div>
<form id="f">
  <input id="msg" placeholder="မေးပါ..." autocomplete="off" autofocus>
  <button id="send">Send</button>
</form>
<script>
const chat = document.getElementById('chat');
const form = document.getElementById('f');
const input = document.getElementById('msg');
const send = document.getElementById('send');

function addMsg(text, who) {
  const d = document.createElement('div');
  d.className = 'msg ' + who;
  d.textContent = text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

form.onsubmit = async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;
  addMsg(msg, 'you');
  input.value = '';
  send.disabled = true;
  try {
    const r = await fetch('/chat', { method: 'POST', body: msg });
    const text = await r.text();
    addMsg(text || '(no response)', 'nyanlin');
  } catch (err) {
    addMsg('(error: ' + err + ')', 'nyanlin');
  }
  send.disabled = false;
  input.focus();
};

addMsg('မင်္ဂလာပါ။ ကျွန်တော် ဉာဏ်လင်း ပါ။ ဘာကူညီပေးရမလဲ?', 'nyanlin');
</script>
</body>
</html>"""


def call_chat(user_input):
    """Call chat_v14 subprocess, extract Nyanlin response."""
    if not os.path.exists(CHAT):
        return f"(error: {CHAT} not found)"
    try:
        result = subprocess.run(
            [CHAT, MODEL],
            input=user_input + "\n",
            capture_output=True,
            text=True,
            timeout=45,
            cwd=os.path.expanduser("~/mmc_publish"),
        )
        # Parse "Nyanlin: ..." from stdout
        # Output format: "You: Nyanlin: <response>"
        # So "Nyanlin:" is NOT at line start
        for line in result.stdout.split("\n"):
            if "Nyanlin:" in line:
                idx = line.find("Nyanlin:")
                return line[idx + len("Nyanlin:"):].strip()
        return "(no response)"
    except subprocess.TimeoutExpired:
        return "(timeout — model slow on this device)"
    except Exception as e:
        return f"(error: {e})"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="replace").strip()
        if not body:
            resp = "(empty input)"
        else:
            # Layer 1: Python KB (fast, easy to edit)
            kb_answer = match_kb(body)
            if kb_answer:
                resp = kb_answer
            else:
                # Layer 2: chat_v14 (MMC rules + model)
                resp = call_chat(body)
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(resp.encode("utf-8"))

    def log_message(self, fmt, *args):
        print(f"[{self.client_address[0]}] {fmt % args}")


if __name__ == "__main__":
    if not os.path.exists(CHAT):
        print(f"ERROR: {CHAT} not found. Run build first.")
        raise SystemExit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: {MODEL} not found.")
        raise SystemExit(1)
    print(f"🐉 Nyanlin Chat Server")
    print(f"   Model: {MODEL}")
    print(f"   Binary: {CHAT}")
    print(f"   URL: http://localhost:{PORT}")
    print(f"   Press Ctrl+C to stop")
    print()
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
