#!/usr/bin/env python3
import asyncio
import os
import sys
import struct
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "proto"))

import websockets
from proto.ipc import ipc_pb2, omgwords_pb2, omgseeks_pb2

DEFAULT_API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
DEFAULT_USERNAME = "blackwing999"

async def check():
    url = "https://woogles.io/api/user_service.AuthenticationService/GetSocketToken"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": DEFAULT_API_KEY,
        "User-Agent": f"TesseraScrabbleEngine/1.0 ({DEFAULT_USERNAME})"
    }
    resp = requests.post(url, headers=headers, json={}, timeout=10)
    data = resp.json()
    token = data["token"]
    cid = data["cid"]

    ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        for _ in range(10):
            frame = await asyncio.wait_for(ws.recv(), timeout=3.0)
            if len(frame) < 3:
                continue
            short, msg_type = struct.unpack_from(">HB", frame, 0)
            payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]
            name = ipc_pb2.MessageType.DESCRIPTOR.values_by_number.get(msg_type)
            name_str = name.name if name else str(msg_type)
            if msg_type == ipc_pb2.LAG_MEASUREMENT:
                continue
            print(f"[RECV] Msg {msg_type} ({name_str}) len={len(payload)}")

            if msg_type == ipc_pb2.ONGOING_GAMES:
                ogs = omgwords_pb2.OngoingGames()
                ogs.ParseFromString(payload)
                my_games = []
                for g in ogs.games:
                    players = [p.nickname for p in g.players]
                    if DEFAULT_USERNAME in players:
                        my_games.append((g.game_id, players))
                print(f"[!] My ongoing games: {my_games}")
                return

if __name__ == "__main__":
    asyncio.run(check())
