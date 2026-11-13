#!/usr/bin/env python3
"""
Test posting an open seek to the Woogles lobby and cancelling it.
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

async def test_lobby_seek():
    auth_data = get_socket_token()
    token = auth_data["token"]
    cid = auth_data["cid"]
    ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
    print(f"[+] Connecting...")

    async with websockets.connect(ws_url) as ws:
        print("[+] Connected! Creating open CSW24 seek in lobby...")
        gr = omgwords_pb2.GameRequest()
        gr.lexicon = "CSW24"
        gr.rules.board_layout_name = "CrosswordGame"
        gr.rules.letter_distribution_name = "english"
        gr.rules.variant_name = "classic"
        gr.initial_time_seconds = 1500
        gr.increment_seconds = 0
        gr.max_overtime_minutes = 1
        gr.challenge_rule = macondo_pb2.ChallengeRule.FIVE_POINT
        gr.game_mode = omgwords_pb2.REAL_TIME
        gr.rating_mode = omgwords_pb2.RATED
        gr.player_vs_bot = False

        sr = omgseeks_pb2.SeekRequest()
        sr.game_request.CopyFrom(gr)
        sr.user_state = omgseeks_pb2.SeekState.READY
        sr.receiver_is_permanent = False
        sr.minimum_rating_range = -500
        sr.maximum_rating_range = 500

        seek_frame = encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr)
        print(f"[+] Sending SEEK_REQUEST ({len(seek_frame)} bytes)...")
        await ws.send(seek_frame)

        # Listen for broadcast
        start = time.time()
        while time.time() - start < 6.0:
            try:
                frame = await asyncio.wait_for(ws.recv(), timeout=1.5)
                if len(frame) < 3:
                    continue
                short, msg_type = struct.unpack_from(">HB", frame, 0)
                payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]

                if msg_type in (ipc_pb2.USER_PRESENCE, ipc_pb2.USER_PRESENCES,
                                ipc_pb2.ONGOING_GAME_EVENT, ipc_pb2.ONGOING_GAMES,
                                ipc_pb2.LAG_MEASUREMENT):
                    continue

                name = ipc_pb2.MessageType.Name(msg_type) if msg_type in ipc_pb2.MessageType.values() else f"Type({msg_type})"
                print(f"[EVENT] {name} ({msg_type}), len={len(payload)}")

                if msg_type == ipc_pb2.SEEK_REQUEST:
                    r = omgseeks_pb2.SeekRequest()
                    r.ParseFromString(payload)
                    print(f"[🔥] SEEK BROADCAST RECEIVED! User={r.user.display_name if r.user else '?'}, ReqID={r.game_request.request_id if r.game_request else '?'}")
                    
                    # Clean up by cancelling the seek
                    if r.game_request and r.game_request.request_id:
                        dsr = omgseeks_pb2.DeclineSeekRequest()
                        dsr.request_id = r.game_request.request_id
                        await ws.send(encode_socket_msg(ipc_pb2.DECLINE_SEEK_REQUEST, dsr))
                        print(f"[+] Cancelled seek {dsr.request_id}")
                    return
            except asyncio.TimeoutError:
                pass

if __name__ == "__main__":
    asyncio.run(test_lobby_seek())
