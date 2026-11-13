#!/usr/bin/env python3
"""
Inspect open lobby seeks on Woogles.io.
"""

import sys
import os
import asyncio
import struct
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lib")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "proto")))

import websockets
from proto.ipc import ipc_pb2, omgwords_pb2, omgseeks_pb2

API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
USERNAME = "blackwing999"

def get_socket_token():
    url = "https://woogles.io/api/user_service.AuthenticationService/GetSocketToken"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": API_KEY,
        "User-Agent": f"TesseraScrabbleEngine/1.0 ({USERNAME})"
    }
    resp = requests.post(url, headers=headers, json={}, timeout=10)
    resp.raise_for_status()
    return resp.json()

async def inspect_lobby():
    auth_data = get_socket_token()
    token = auth_data["token"]
    cid = auth_data["cid"]
    ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
    print(f"[+] Connecting...")

    async with websockets.connect(ws_url) as ws:
        for _ in range(10):
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=4.0)
                if len(frame) < 3:
                    continue
                short, msg_type = struct.unpack_from(">HB", frame, 0)
                if msg_type == ipc_pb2.SEEK_REQUESTS:
                    payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]
                    sr = omgseeks_pb2.SeekRequests()
                    sr.ParseFromString(payload)
                    print(f"[+] Found {len(sr.requests)} open seeks in lobby:")
                    for req in sr.requests:
                        gr = req.game_request
                        user = req.user.display_name if req.user else "Anonymous"
                        rec = req.receiving_user.display_name if req.receiving_user else "Anyone"
                        print(f"    - From: {user} -> {rec} | Lexicon: {gr.lexicon} | Time: {gr.initial_time_seconds//60}m | Rated: {gr.rating_mode == omgwords_pb2.RATED} | Bot: {gr.player_vs_bot}")
                    return
            except asyncio.TimeoutError:
                break

if __name__ == "__main__":
    asyncio.run(inspect_lobby())
