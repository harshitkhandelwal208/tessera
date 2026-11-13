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
from proto.ipc import ipc_pb2, omgwords_pb2, omgseeks_pb2, errors_pb2
from proto.vendored.macondo import macondo_pb2

DEFAULT_API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
DEFAULT_USERNAME = "blackwing999"

def encode_socket_msg(msg_type: int, pb_obj) -> bytes:
    payload = pb_obj.SerializeToString()
    pkt_len = len(payload) + 1
    return struct.pack(">HB", pkt_len, msg_type) + payload

async def test_bot_seek():
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
        # Drain init frames
        for _ in range(5):
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.2)
            except asyncio.TimeoutError:
                break

        gr = omgwords_pb2.GameRequest()
        gr.lexicon = "CSW24"
        gr.rules.board_layout_name = "CrosswordGame"
        gr.rules.letter_distribution_name = "english"
        gr.rules.variant_name = "classic"
        gr.initial_time_seconds = 15 * 60
        gr.increment_seconds = 0
        gr.max_overtime_minutes = 1
        gr.challenge_rule = macondo_pb2.ChallengeRule.FIVE_POINT
        gr.game_mode = omgwords_pb2.REAL_TIME
        gr.rating_mode = omgwords_pb2.RATED
        gr.player_vs_bot = True
        gr.bot_type = macondo_pb2.BotRequest.HASTY_BOT

        sr = omgseeks_pb2.SeekRequest()
        sr.game_request.CopyFrom(gr)
        sr.user_state = omgseeks_pb2.SeekState.READY
        sr.receiver_is_permanent = False

        print("[*] Sending bot seek (receiver_is_permanent = False)...")
        await ws.send(encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr))

        for _ in range(10):
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=3.0)
            except asyncio.TimeoutError:
                print("[*] Waiting...")
                continue
            if len(frame) < 3:
                continue
            short, msg_type = struct.unpack_from(">HB", frame, 0)
            payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]
            name = ipc_pb2.MessageType.DESCRIPTOR.values_by_number.get(msg_type)
            name_str = name.name if name else str(msg_type)
            if msg_type == ipc_pb2.LAG_MEASUREMENT:
                continue
            print(f"[RECV] Msg {msg_type} ({name_str}) len={len(payload)}")

            if msg_type == ipc_pb2.NEW_GAME_EVENT:
                nge = omgwords_pb2.NewGameEvent()
                nge.ParseFromString(payload)
                print(f"[🔥 SUCCESS!] NewGameEvent GameID = {nge.game_id}")
                return nge.game_id
            elif msg_type == ipc_pb2.ERROR_MESSAGE:
                err = errors_pb2.ErrorMessage()
                err.ParseFromString(payload)
                print(f"[-] SERVER ERROR: {err.message}")
                return None

if __name__ == "__main__":
    asyncio.run(test_bot_seek())
