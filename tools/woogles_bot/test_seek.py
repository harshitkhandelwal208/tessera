#!/usr/bin/env python3
"""
Test seeking a game against HastyBot with receiving_user set.
"""

import sys
import os
import asyncio
import time
import requests
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "lib")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "proto")))

import websockets
from proto.ipc import ipc_pb2, omgwords_pb2, omgseeks_pb2
from proto.vendored.macondo import macondo_pb2

API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
USERNAME = "blackwing999"

def encode_socket_msg(msg_type: int, pb_obj) -> bytes:
    payload = pb_obj.SerializeToString()
    pkt_len = len(payload) + 1
    header = struct.pack(">HB", pkt_len, msg_type)
    return header + payload

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

async def test_bot_seek():
    auth_data = get_socket_token()
    token = auth_data["token"]
    cid = auth_data["cid"]
    ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
    print(f"[+] Connecting to WebSocket: cid={cid}...")

    headers = {
        "User-Agent": f"TesseraScrabbleEngine/1.0 ({USERNAME})"
    }

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        print("[+] WebSocket connected!")

        # Drain initial frames for 0.5s
        start_drain = time.time()
        while time.time() - start_drain < 0.5:
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        print("[+] Creating Match Request for HastyBot (CSW24, Rated)...")
        gr = omgwords_pb2.GameRequest()
        gr.lexicon = "CSW24"
        gr.rules.board_layout_name = "CrosswordGame"
        gr.rules.letter_distribution_name = "english"
        gr.rules.variant_name = "classic"
        gr.initial_time_seconds = 1200  # 20 minutes
        gr.increment_seconds = 0
        gr.max_overtime_minutes = 1
        gr.challenge_rule = macondo_pb2.ChallengeRule.VOID
        gr.game_mode = omgwords_pb2.REAL_TIME
        gr.rating_mode = omgwords_pb2.RATED
        gr.player_vs_bot = True
        gr.bot_type = macondo_pb2.BotRequest.HASTY_BOT

        sr = omgseeks_pb2.SeekRequest()
        sr.game_request.CopyFrom(gr)
        sr.user_state = omgseeks_pb2.SeekState.READY
        sr.receiver_is_permanent = True
        sr.receiving_user.display_name = "HastyBot"
        sr.receiving_user.user_id = "dJHCp6DkBPZKjBNxYAPkiJ"
        sr.minimum_rating_range = -500
        sr.maximum_rating_range = 500

        seek_frame = encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr)
        print(f"[+] Sending SEEK_REQUEST ({len(seek_frame)} bytes)...")
        await ws.send(seek_frame)

        # Wait up to 10 seconds for response
        start_wait = time.time()
        while time.time() - start_wait < 10.0:
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                if len(frame) < 3:
                    continue
                short, msg_type = struct.unpack_from(">HB", frame, 0)
                payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]

                if msg_type in (ipc_pb2.USER_PRESENCE, ipc_pb2.USER_PRESENCES,
                                ipc_pb2.ONGOING_GAME_EVENT, ipc_pb2.ONGOING_GAMES,
                                ipc_pb2.LAG_MEASUREMENT, ipc_pb2.GAME_DELETION):
                    continue

                name = ipc_pb2.MessageType.Name(msg_type) if msg_type in ipc_pb2.MessageType.values() else f"Type({msg_type})"
                print(f"[⭐ EVENT] {name} ({msg_type}), len={len(payload)}")

                if msg_type == ipc_pb2.ERROR_MESSAGE:
                    err_msg = ipc_pb2.ErrorMessage()
                    err_msg.ParseFromString(payload)
                    print(f"[-] SERVER ERROR: {err_msg.message}")
                    return

                elif msg_type == ipc_pb2.NEW_GAME_EVENT:
                    nge = omgwords_pb2.NewGameEvent()
                    nge.ParseFromString(payload)
                    print(f"[🏆] NEW GAME CREATED: GameID = {nge.game_id}")
                    
                    rfg = omgwords_pb2.ReadyForGame()
                    rfg.game_id = nge.game_id
                    await ws.send(encode_socket_msg(ipc_pb2.READY_FOR_GAME, rfg))
                    print(f"[+] Sent READY_FOR_GAME for {nge.game_id}")

                elif msg_type == ipc_pb2.GAME_HISTORY_REFRESHER:
                    ghr = omgwords_pb2.GameHistoryRefresher()
                    ghr.ParseFromString(payload)
                    print(f"[🎲] GAME START: UID = {ghr.history.uid}")
                    print(f"    Lexicon: {ghr.history.lexicon}")
                    print(f"    Players: {[p.nickname for p in ghr.history.players]}")
                    print(f"    Racks: {list(ghr.history.last_known_racks)}")
                    print(f"    Turns: {len(ghr.history.events)}")
                    return
            except asyncio.TimeoutError:
                print(f"[*] Waiting... ({int(time.time() - start_wait)}s)")

if __name__ == "__main__":
    asyncio.run(test_bot_seek())
