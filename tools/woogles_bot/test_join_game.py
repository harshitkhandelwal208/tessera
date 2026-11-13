#!/usr/bin/env python3
"""
Diagnostic: Connect to path=/game/{game_id} and listen for GAME_HISTORY_REFRESHER.
"""
import asyncio
import os
import sys
import struct
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "proto"))

import websockets
from proto.ipc import ipc_pb2, omgwords_pb2
from proto.vendored.macondo import macondo_pb2

DEFAULT_API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
DEFAULT_USERNAME = "blackwing999"
GAME_ID = sys.argv[1] if len(sys.argv) > 1 else "8y6AUhpviQ"

def encode_socket_msg(msg_type: int, pb_obj) -> bytes:
    payload = pb_obj.SerializeToString()
    pkt_len = len(payload) + 1
    return struct.pack(">HB", pkt_len, msg_type) + payload

async def test_game_socket():
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
    print(f"[+] Got token & CID: {cid}")

    game_path = f"/game/{GAME_ID}"
    ws_url = f"wss://woogles.io/ws?token={token}&path={game_path}&cid={cid}"
    print(f"[*] Connecting to {ws_url[:55]}...")

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        print("[+] Connected to game realm!")

        # Send READY_FOR_GAME
        rfg = omgwords_pb2.ReadyForGame()
        rfg.game_id = GAME_ID
        await ws.send(encode_socket_msg(ipc_pb2.READY_FOR_GAME, rfg))
        print(f"[+] Sent READY_FOR_GAME for {GAME_ID}")

        for _ in range(10):
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                print("[*] Waiting frame...")
                continue

            if len(frame) < 3:
                continue
            short, msg_type = struct.unpack_from(">HB", frame, 0)
            payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]

            msg_name = ipc_pb2.MessageType.DESCRIPTOR.values_by_number.get(msg_type)
            name_str = msg_name.name if msg_name else str(msg_type)

            if msg_type == ipc_pb2.LAG_MEASUREMENT:
                await ws.send(frame)
                continue

            print(f"[RECV in game realm] Msg {msg_type} ({name_str}), payload len: {len(payload)}")
            if msg_type == ipc_pb2.GAME_HISTORY_REFRESHER:
                ghr = omgwords_pb2.GameHistoryRefresher()
                ghr.ParseFromString(payload)
                hist = ghr.history
                print(f"[+] History: {len(hist.events)} events, play_state: {hist.play_state}")
                for idx, p in enumerate(hist.players):
                    rack = hist.last_known_racks[idx] if len(hist.last_known_racks) > idx else ""
                    print(f"    Player {idx}: {p.nickname} (rack: '{rack}')")
                return
            elif msg_type == ipc_pb2.GAME_ENDED_EVENT:
                gee = omgwords_pb2.GameEndedEvent()
                gee.ParseFromString(payload)
                print(f"[!] Game ended: winner={gee.winner}, scores={gee.scores}")
                return

if __name__ == "__main__":
    asyncio.run(test_game_socket())
