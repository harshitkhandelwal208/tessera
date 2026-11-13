#!/usr/bin/env python3
"""
Test WebSocket connection to woogles.io using the GetSocketToken RPC and websockets library.
"""

import sys
import os
import asyncio
import json
import requests
import struct

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
    data = resp.json()
    print(f"[+] Obtained socket token: cid={data.get('cid')}")
    return data

async def test_ws():
    auth_data = get_socket_token()
    token = auth_data["token"]
    cid = auth_data["cid"]
    ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
    print(f"[+] Connecting to WebSocket: {ws_url[:60]}...")

    headers = {
        "User-Agent": f"TesseraScrabbleEngine/1.0 ({USERNAME})"
    }

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        print("[+] WebSocket connected successfully!")
        
        # Wait for initial server messages (e.g. USER_PRESENCES, SEEK_REQUESTS, etc.)
        for i in range(3):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(msg, bytes):
                    print(f"[+] Received binary frame of {len(msg)} bytes")
                    # Parse frame header
                    if len(msg) >= 3:
                        short, msg_type = struct.unpack_from(">HB", msg, 0)
                        msg_name = ipc_pb2.MessageType.Name(msg_type) if msg_type in ipc_pb2.MessageType.values() else f"UNKNOWN({msg_type})"
                        print(f"    Message Type: {msg_name} ({msg_type}), Length: {short}")
                else:
                    print(f"[+] Received text frame: {msg[:100]}")
            except asyncio.TimeoutError:
                print("[-] Timeout waiting for next message")
                break

    print("[+] Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_ws())
