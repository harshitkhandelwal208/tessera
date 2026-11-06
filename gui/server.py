#!/usr/bin/env python3
"""
Tessera Web GUI Server
Bridges the HTTP/JSON Web Frontend with the native C Tessera Scrabble Engine.
Zero external dependencies (uses standard library http.server + subprocess).
"""

import sys
import os
import json
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PORT = 8080
ENGINE_BIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tessera"))
WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "web"))

class EngineBridge:
    def __init__(self, binary_path):
        self.binary_path = binary_path
        self.lock = threading.Lock()
        self.process = None
        self._start_engine()

    def _start_engine(self):
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
        self.process = subprocess.Popen(
            [self.binary_path, "server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def send_command(self, cmd: str) -> dict:
        with self.lock:
            try:
                if not self.process or self.process.poll() is not None:
                    self._start_engine()
                self.process.stdin.write(cmd.strip() + "\n")
                self.process.stdin.flush()
                while True:
                    response_line = self.process.stdout.readline()
                    if not response_line:
                        return {"status": "error", "message": "Engine closed pipe"}
                    line_str = response_line.strip()
                    if line_str.startswith("{") and line_str.endswith("}"):
                        return json.loads(line_str)
            except Exception as e:
                self._start_engine()
                return {"status": "error", "message": str(e)}

engine = None

class ScrabbleHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed.path, parsed.query)
        else:
            super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        data = json.loads(body) if body else {}

        if parsed.path.startswith("/api/"):
            self._handle_api_post(parsed.path, data)
        else:
            self.send_error(404, "Unknown API endpoint")

    def _handle_api(self, path, query_str):
        params = urllib.parse.parse_qs(query_str)
        if path == "/api/state":
            res = engine.send_command("GET_STATE")
            self._send_json(res)
        elif path == "/api/hint":
            res = engine.send_command("HINT")
            self._send_json(res)
        elif path == "/api/ai_move":
            res = engine.send_command("AI_MOVE")
            self._send_json(res)
        elif path == "/api/opponent_move":
            bot_type = params.get("bot_type", ["6"])[0]
            res = engine.send_command(f"OPPONENT_MOVE {bot_type}")
            self._send_json(res)
        elif path == "/api/new_game":
            res = engine.send_command("NEW_GAME")
            self._send_json(res)
        elif path == "/api/check_word":
            word = params.get("word", [""])[0]
            res = engine.send_command(f"CHECK_WORD {word}")
            self._send_json(res)
        else:
            self.send_error(404, "API endpoint not found")

    def _handle_api_post(self, path, data):
        if path == "/api/play_move":
            row = data.get("row", 0)
            col = data.get("col", 0)
            dir_str = data.get("dir", "H")
            word = data.get("word", "").upper()
            cmd = f"PLAY_MOVE {row} {col} {dir_str} {word}"
            res = engine.send_command(cmd)
            self._send_json(res)
        elif path == "/api/exchange":
            tiles = data.get("tiles", "").upper()
            res = engine.send_command(f"EXCHANGE {tiles}")
            self._send_json(res)
        elif path == "/api/pass":
            res = engine.send_command("PASS")
            self._send_json(res)
        elif path == "/api/opponent_move":
            bot_type = data.get("bot_type", 6)
            res = engine.send_command(f"OPPONENT_MOVE {bot_type}")
            self._send_json(res)
        else:
            self.send_error(404, "API endpoint not found")

    def _send_json(self, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

def main():
    global engine
    if not os.path.isfile(ENGINE_BIN):
        print(f"Error: Engine binary not found at {ENGINE_BIN}")
        print("Please run `make` first to build the native engine.")
        sys.exit(1)

    print(f"[SERVER] Starting native Tessera engine process ({ENGINE_BIN})...")
    engine = EngineBridge(ENGINE_BIN)

    # Initial test
    st = engine.send_command("GET_STATE")
    if st.get("status") != "ok":
        print(f"Error communicating with engine: {st}")
        sys.exit(1)
    print(f"[SERVER] Engine ready! Tile bag: {st.get('bag_count')} tiles.")

    server_address = ("", PORT)
    httpd = HTTPServer(server_address, ScrabbleHTTPHandler)
    print(f"[SERVER] Serving Tessera Scrabble GUI at http://localhost:{PORT}")
    print("[SERVER] Press Ctrl+C to terminate.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down...")
    finally:
        if engine and engine.process:
            engine.process.terminate()

if __name__ == "__main__":
    main()
