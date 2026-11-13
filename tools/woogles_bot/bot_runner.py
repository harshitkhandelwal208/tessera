#!/usr/bin/env python3
"""
Tessera Woogles Bot Runner
Autonomous Tournament Runner for live official rated play on Woogles.io.
Connects via WebSocket, manages game lifecycle, challenges HastyBot or
accepts lobby seeks, reconstructs board states, and executes optimal moves via libtessera.so.
"""

import argparse
import asyncio
import os
import sys
import time
import struct
import json
from typing import Dict, List, Optional, Tuple, Any
import requests

# Set up local paths for libraries and protobufs
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "lib"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "proto"))
sys.path.insert(0, PROJECT_ROOT)

import websockets
from proto.ipc import ipc_pb2, omgwords_pb2, omgseeks_pb2, errors_pb2
from proto.vendored.macondo import macondo_pb2

from tools.woogles_bot.engine import TesseraEngine

DEFAULT_API_KEY = "BycgN8zzivRF3QVkzscabgFErcTuXAi6ZQAYsT8RDwvm"
DEFAULT_USERNAME = "blackwing999"

BOT_CODES = {
    "HastyBot": macondo_pb2.BotRequest.HASTY_BOT,
    "BestBot": macondo_pb2.BotRequest.SIMMING_BOT,
    "STEEBot": macondo_pb2.BotRequest.LEVEL4_PROBABILISTIC,
    "BetterBot": macondo_pb2.BotRequest.LEVEL3_PROBABILISTIC,
    "BasicBot": macondo_pb2.BotRequest.LEVEL2_PROBABILISTIC,
    "BeginnerBot": macondo_pb2.BotRequest.LEVEL1_PROBABILISTIC,
}

def encode_socket_msg(msg_type: int, pb_obj) -> bytes:
    payload = pb_obj.SerializeToString()
    pkt_len = len(payload) + 1
    header = struct.pack(">HB", pkt_len, msg_type)
    return header + payload

class WooglesBot:
    def __init__(self,
                 username: str = DEFAULT_USERNAME,
                 api_key: str = DEFAULT_API_KEY,
                 lexicon: str = "CSW24",
                 delib_steps: int = 25,
                 time_minutes: int = 15,
                 rated: bool = True,
                 auto_accept: bool = True,
                 target_games: int = 1,
                 vs_bot: Optional[str] = "HastyBot",
                 use_lobby: bool = False):
        self.username = username.strip()
        self.api_key = api_key.strip()
        self.lexicon = lexicon
        self.delib_steps = delib_steps
        self.time_minutes = time_minutes
        self.rated = rated
        self.auto_accept = auto_accept
        self.target_games = target_games
        self.vs_bot = vs_bot
        self.use_lobby = use_lobby

        self.engine = TesseraEngine()
        self.games_completed = 0
        self.active_seek_id: Optional[str] = None
        self.my_seeks = set()

    def get_socket_token(self) -> Dict[str, Any]:
        url = "https://woogles.io/api/user_service.AuthenticationService/GetSocketToken"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "User-Agent": f"TesseraScrabbleEngine/1.0 ({self.username})"
        }
        delays = [1.5, 3.0, 6.0, 10.0]
        for attempt in range(len(delays)):
            try:
                resp = requests.post(url, headers=headers, json={}, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == len(delays) - 1:
                    raise
                print(f"[*] GetSocketToken retry {attempt+1}/{len(delays)} after error: {e}", flush=True)
                time.sleep(delays[attempt])
        raise RuntimeError("Failed to obtain socket token")

    def get_official_ratings(self) -> Dict[str, Any]:
        url = "https://woogles.io/api/user_service.ProfileService/GetRatings"
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "User-Agent": f"TesseraScrabbleEngine/1.0 ({self.username})"
        }
        try:
            resp = requests.post(url, headers=headers, json={"username": self.username}, timeout=10)
            resp.raise_for_status()
            raw = resp.json().get("json", "{}")
            return json.loads(raw).get("Data", {})
        except Exception:
            return {}

    def display_rating_card(self, label: str = "OFFICIAL WOOGLES RATING CARD"):
        ratings = self.get_official_ratings()
        print("\n" + "=" * 68, flush=True)
        print(f"             {label}             ", flush=True)
        print("=" * 68, flush=True)
        print(f"  Player Account:  @{self.username}", flush=True)
        if ratings:
            for key, info in sorted(ratings.items()):
                r = info.get("r", 1500)
                rd = info.get("rd", 350)
                print(f"  Category: {key:<28} Rating: {r:6.1f} (±{rd:.1f} RD)", flush=True)
        else:
            print("  No rated games recorded yet.", flush=True)
        print("=" * 68 + "\n", flush=True)

    def rebuild_board(self, events) -> Tuple[List[List[str]], List[List[bool]]]:
        board = [[' ' for _ in range(15)] for _ in range(15)]
        blanks = [[False for _ in range(15)] for _ in range(15)]

        for evt in events:
            if evt.type == 0 and evt.position and evt.played_tiles:  # TILE_PLACEMENT_MOVE
                pos = evt.position.strip()
                tiles = evt.played_tiles

                if pos[0].isdigit():
                    col_char = pos[-1].upper()
                    row_num = int(pos[:-1])
                    row = row_num - 1
                    col = ord(col_char) - ord('A')
                    horiz = True
                else:
                    col_char = pos[0].upper()
                    row_num = int(pos[1:])
                    row = row_num - 1
                    col = ord(col_char) - ord('A')
                    horiz = False

                for k, ch in enumerate(tiles):
                    r = row if horiz else row + k
                    c = col + k if horiz else col
                    if 0 <= r < 15 and 0 <= c < 15:
                        if ch != '.':
                            board[r][c] = ch.upper()
                            blanks[r][c] = ch.islower()

        return board, blanks

    def format_played_tiles(self, board_before: List[List[str]], move: Dict[str, Any]) -> str:
        row = move["row"]
        col = move["col"]
        horiz = move["horizontal"]
        word = move["word"]

        placed_map = {}
        for p in move["placed"]:
            placed_map[(p["row"], p["col"])] = p

        tile_chars = []
        for k, ch in enumerate(word):
            r = row if horiz else row + k
            c = col + k if horiz else col
            if (r, c) in placed_map:
                pt = placed_map[(r, c)]
                tile_chars.append(ch.lower() if pt["is_blank"] else ch.upper())
            else:
                tile_chars.append('.')

        return "".join(tile_chars)

    def calculate_game_metrics(self, history, our_index: int) -> Tuple[int, int, int, int]:
        opp_index = 1 - our_index
        our_score = 0
        opp_score = 0
        total_tiles_played = 0

        for evt in history.events:
            if evt.player_index == our_index and evt.cumulative > 0:
                our_score = evt.cumulative
            elif evt.player_index == opp_index and evt.cumulative > 0:
                opp_score = evt.cumulative
            if evt.type == 0:  # TILE_PLACEMENT_MOVE
                for ch in evt.played_tiles:
                    if ch != '.':
                        total_tiles_played += 1

        our_rack_len = len(history.last_known_racks[our_index]) if len(history.last_known_racks) > our_index else 7
        opp_rack_len = 7
        bag_count = max(0, 100 - total_tiles_played - our_rack_len - opp_rack_len)
        score_diff = our_score - opp_score

        return bag_count, our_score, opp_score, score_diff

    def get_player_on_turn(self, history) -> int:
        on_turn = 0
        for evt in history.events:
            if evt.type in (0, 2, 4, 8):  # plays that consume a turn
                on_turn = (on_turn + 1) % len(history.players)
        return on_turn

    async def play_game(self, game_id: str) -> bool:
        """
        Connects to path=/game/{game_id} and plays out the game to completion.
        Returns True if game finished successfully.
        """
        print("\n" + "=" * 70, flush=True)
        print(f"[*] Entering Game Realm: /game/{game_id}", flush=True)
        print("=" * 70, flush=True)

        auth_data = self.get_socket_token()
        token = auth_data["token"]
        cid = auth_data["cid"]
        ws_url = f"wss://woogles.io/ws?token={token}&path=/game/{game_id}&cid={cid}"
        headers = {"User-Agent": f"TesseraScrabbleEngine/1.0 ({self.username})"}

        self.engine.reset_tracker()
        last_turn_processed = -1

        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            print(f"[+] Connected to game realm socket for {game_id}", flush=True)

            # Send READY_FOR_GAME
            rfg = omgwords_pb2.ReadyForGame()
            rfg.game_id = game_id
            await ws.send(encode_socket_msg(ipc_pb2.READY_FOR_GAME, rfg))
            print(f"[+] Sent READY_FOR_GAME signal for {game_id}", flush=True)

            while True:
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue

                if len(frame) < 3:
                    continue

                short, msg_type = struct.unpack_from(">HB", frame, 0)
                payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]

                if msg_type == ipc_pb2.LAG_MEASUREMENT:
                    continue

                if msg_type == ipc_pb2.GAME_HISTORY_REFRESHER:
                    ghr = omgwords_pb2.GameHistoryRefresher()
                    ghr.ParseFromString(payload)
                    hist = ghr.history

                    if hist.play_state == macondo_pb2.PlayState.GAME_OVER:
                        print(f"[*] Game play state is GAME_OVER.", flush=True)
                        continue

                    our_idx = -1
                    for idx, p in enumerate(hist.players):
                        if p.nickname.lower() == self.username.lower():
                            our_idx = idx
                            break

                    if our_idx == -1:
                        print(f"[-] Could not find username {self.username} in players list!", flush=True)
                        continue

                    opp_idx = 1 - our_idx
                    opp_name = hist.players[opp_idx].nickname
                    num_events = len(hist.events)
                    turn_idx = self.get_player_on_turn(hist)

                    if turn_idx == our_idx:
                        if num_events == last_turn_processed:
                            continue

                        our_rack = hist.last_known_racks[our_idx] if len(hist.last_known_racks) > our_idx else ""
                        if not our_rack:
                            continue

                        board_grid, blanks_grid = self.rebuild_board(hist.events)
                        bag, our_sc, opp_sc, diff = self.calculate_game_metrics(hist, our_idx)

                        turn_number = (num_events // 2) + 1
                        print(f"\n[TURN {turn_number}] Tessera (@{self.username}) vs @{opp_name}", flush=True)
                        print(f"    Current Scores: {self.username} {our_sc} - {opp_sc} {opp_name} (Spread: {diff:+d})", flush=True)
                        print(f"    Rack: [{our_rack}] | Bag: {bag} tiles remaining", flush=True)
                        print(f"    Deliberating with Belief-Lattice ({self.delib_steps} steps)...", flush=True)

                        start_t = time.time()
                        move = self.engine.compute_move(
                            board_grid=board_grid,
                            blanks_grid=blanks_grid,
                            rack=our_rack,
                            bag_count=bag,
                            score_diff=diff,
                            delib_steps=self.delib_steps
                        )
                        calc_ms = (time.time() - start_t) * 1000.0

                        cge = omgwords_pb2.ClientGameplayEvent()
                        cge.game_id = game_id

                        if move["type"] == "PLAY":
                            cge.type = omgwords_pb2.ClientGameplayEvent.TILE_PLACEMENT
                            cge.position_coords = move["coord"]
                            tiles_str = self.format_played_tiles(board_grid, move)
                            cge.tiles = tiles_str
                            print(f"    >>> PLAY: {move['word']} at {move['coord']} (+{move['score']} pts, latency: {calc_ms:.1f}ms)", flush=True)
                        elif move["type"] == "EXCHANGE":
                            cge.type = omgwords_pb2.ClientGameplayEvent.EXCHANGE
                            cge.tiles = move.get("tiles", "")
                            print(f"    >>> EXCHANGE: [{cge.tiles}] (latency: {calc_ms:.1f}ms)", flush=True)
                        else:
                            cge.type = omgwords_pb2.ClientGameplayEvent.PASS
                            print(f"    >>> PASS (latency: {calc_ms:.1f}ms)", flush=True)

                        await ws.send(encode_socket_msg(ipc_pb2.CLIENT_GAMEPLAY_EVENT, cge))
                        last_turn_processed = num_events
                    else:
                        print(f"[*] Opponent @{opp_name} is on turn...", flush=True)

                elif msg_type == ipc_pb2.GAME_ENDED_EVENT:
                    gee = omgwords_pb2.GameEndedEvent()
                    gee.ParseFromString(payload)
                    end_desc = omgwords_pb2.GameEndReason.DESCRIPTOR.values_by_number.get(gee.end_reason)
                    end_str = end_desc.name if end_desc else str(gee.end_reason)

                    is_valid_game = (gee.end_reason not in (
                        omgwords_pb2.GameEndReason.ABORTED,
                        omgwords_pb2.GameEndReason.CANCELLED
                    ))
                    if is_valid_game:
                        self.games_completed += 1

                    print("\n" + "=" * 70, flush=True)
                    print(f"[🏁 MATCH FINISHED] Official Result:", flush=True)
                    print(f"  Game ID: {game_id}", flush=True)
                    for user, sc in gee.scores.items():
                        delta = gee.rating_deltas.get(user, 0)
                        new_r = gee.new_ratings.get(user, 0)
                        print(f"    @{user:<15}: {sc:3d} pts | Rating: {new_r} ({delta:+d})", flush=True)
                    print(f"  Official Winner: @{gee.winner}", flush=True)
                    print(f"  End Reason:      {end_str}", flush=True)
                    print("=" * 70 + "\n", flush=True)

                    if is_valid_game:
                        self.display_rating_card("UPDATED OFFICIAL WOOGLES RATING CARD")
                    else:
                        print("[!] Game was aborted/cancelled before play. Re-seeking active match...", flush=True)
                    return is_valid_game

                elif msg_type == ipc_pb2.ERROR_MESSAGE:
                    err = errors_pb2.ErrorMessage()
                    err.ParseFromString(payload)
                    print(f"[-] SERVER ERROR in game realm: {err.message}", flush=True)

    async def run(self):
        print("=" * 80, flush=True)
        print("           TESSERA AI -- OFFICIAL WOOGLES.IO TOURNAMENT BOT CLIENT          ", flush=True)
        print("=" * 80, flush=True)
        print(f"  Account:       @{self.username}", flush=True)
        print(f"  Lexicon:       {self.lexicon}", flush=True)
        print(f"  Mode:          {'RATED (Audited Glicko-2)' if self.rated else 'CASUAL'}", flush=True)
        print(f"  Deliberation:  {self.delib_steps} lattice steps", flush=True)
        print(f"  Target Games:  {self.target_games}", flush=True)
        mode_desc = "Open Lobby Matching" if self.use_lobby else f"@{self.vs_bot}"
        print(f"  Match Opponent: {mode_desc}", flush=True)
        print(f"  Engine:        libtessera.so (Distributional Belief-Lattice C11)", flush=True)
        print("=" * 80, flush=True)

        self.display_rating_card("INITIAL RATINGS AT SESSION START")

        while self.games_completed < self.target_games:
            auth_data = self.get_socket_token()
            token = auth_data["token"]
            cid = auth_data["cid"]
            ws_url = f"wss://woogles.io/ws?token={token}&path=/&cid={cid}"
            headers = {"User-Agent": f"TesseraScrabbleEngine/1.0 ({self.username})"}

            print(f"[*] Connecting to Lobby: {ws_url[:45]}... (CID: {cid})", flush=True)

            async with websockets.connect(ws_url, additional_headers=headers) as ws:
                print("[+] Connected to Lobby successfully!\n", flush=True)

                # Drain lobby initialization messages for 1.0s so NATS connection is established
                init_drain = time.time()
                while time.time() - init_drain < 1.0:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=0.2)
                    except asyncio.TimeoutError:
                        break

                if not self.use_lobby and self.vs_bot:
                    bot_code = BOT_CODES.get(self.vs_bot, macondo_pb2.BotRequest.HASTY_BOT)
                    print(f"[*] Issuing challenge to @{self.vs_bot} ({self.lexicon}, {self.time_minutes}m, {'RATED' if self.rated else 'CASUAL'})...", flush=True)
                    gr = omgwords_pb2.GameRequest()
                    gr.lexicon = self.lexicon
                    gr.rules.board_layout_name = "CrosswordGame"
                    gr.rules.letter_distribution_name = "english"
                    gr.rules.variant_name = "classic"
                    gr.initial_time_seconds = self.time_minutes * 60
                    gr.increment_seconds = 0
                    gr.max_overtime_minutes = 1
                    gr.challenge_rule = macondo_pb2.ChallengeRule.FIVE_POINT
                    gr.game_mode = omgwords_pb2.REAL_TIME
                    gr.rating_mode = omgwords_pb2.RATED if self.rated else omgwords_pb2.CASUAL
                    gr.player_vs_bot = True
                    gr.bot_type = bot_code

                    sr = omgseeks_pb2.SeekRequest()
                    sr.game_request.CopyFrom(gr)
                    sr.user_state = omgseeks_pb2.SeekState.READY
                    sr.receiver_is_permanent = True

                    await ws.send(encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr))
                else:
                    print(f"[*] Posting open rated {self.lexicon} challenge to Woogles lobby...", flush=True)
                    gr = omgwords_pb2.GameRequest()
                    gr.lexicon = self.lexicon
                    gr.rules.board_layout_name = "CrosswordGame"
                    gr.rules.letter_distribution_name = "english"
                    gr.rules.variant_name = "classic"
                    gr.initial_time_seconds = self.time_minutes * 60
                    gr.increment_seconds = 0
                    gr.max_overtime_minutes = 1
                    gr.challenge_rule = macondo_pb2.ChallengeRule.FIVE_POINT
                    gr.game_mode = omgwords_pb2.REAL_TIME
                    gr.rating_mode = omgwords_pb2.RATED if self.rated else omgwords_pb2.CASUAL
                    gr.player_vs_bot = False

                    sr = omgseeks_pb2.SeekRequest()
                    sr.game_request.CopyFrom(gr)
                    sr.user_state = omgseeks_pb2.SeekState.READY
                    sr.receiver_is_permanent = False
                    sr.minimum_rating_range = -600
                    sr.maximum_rating_range = 600

                    await ws.send(encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr))

                matched_game_id = None
                last_seek_refresh = time.time()

                while not matched_game_id:
                    try:
                        frame = await asyncio.wait_for(ws.recv(), timeout=2.0)
                    except asyncio.TimeoutError:
                        if self.use_lobby and time.time() - last_seek_refresh > 25.0:
                            # Re-post open seek if sitting in lobby
                            print(f"[*] Refreshing open seek in lobby...", flush=True)
                            await ws.send(encode_socket_msg(ipc_pb2.SEEK_REQUEST, sr))
                            last_seek_refresh = time.time()
                        continue

                    if len(frame) < 3:
                        continue

                    short, msg_type = struct.unpack_from(">HB", frame, 0)
                    payload = frame[3:2+short] if len(frame) >= 2+short else frame[3:]

                    if msg_type == ipc_pb2.LAG_MEASUREMENT:
                        continue

                    name_desc = ipc_pb2.MessageType.DESCRIPTOR.values_by_number.get(msg_type)
                    name_str = name_desc.name if name_desc else str(msg_type)
                    if msg_type not in (ipc_pb2.USER_PRESENCE, ipc_pb2.ONGOING_GAME_EVENT):
                        print(f"[*] Lobby recv: Msg {msg_type} ({name_str}), len={len(payload)}", flush=True)

                    if msg_type == ipc_pb2.NEW_GAME_EVENT:
                        nge = omgwords_pb2.NewGameEvent()
                        nge.ParseFromString(payload)
                        matched_game_id = nge.game_id
                        print(f"\n[🔥 MATCH PAIRED!] Official Game ID: {matched_game_id}", flush=True)
                        break

                    elif msg_type == ipc_pb2.SEEK_REQUESTS and self.auto_accept:
                        srs = omgseeks_pb2.SeekRequests()
                        srs.ParseFromString(payload)
                        for req in srs.requests:
                            user = req.user.display_name if req.user else ""
                            gr = req.game_request
                            if not gr or not gr.request_id:
                                continue
                            if user == self.username:
                                self.my_seeks.add(gr.request_id)
                                continue
                            if gr.lexicon == self.lexicon and not req.receiving_user.display_name and (not self.rated or gr.rating_mode == omgwords_pb2.RATED):
                                print(f"[🔥 LOBBY SEEK DETECTED] Accepting open seek from @{user}...", flush=True)
                                sa = omgseeks_pb2.SoughtGameProcessEvent()
                                sa.request_id = gr.request_id
                                await ws.send(encode_socket_msg(ipc_pb2.SOUGHT_GAME_PROCESS_EVENT, sa))
                                break

                    elif msg_type == ipc_pb2.SEEK_REQUEST:
                        req = omgseeks_pb2.SeekRequest()
                        req.ParseFromString(payload)
                        user = req.user.display_name if req.user else ""
                        gr = req.game_request
                        if gr and gr.request_id:
                            if user == self.username:
                                self.my_seeks.add(gr.request_id)
                                print(f"[+] Our seek registered in lobby: ID={gr.request_id}", flush=True)
                            elif self.auto_accept and gr.lexicon == self.lexicon and not req.receiving_user.display_name and (not self.rated or gr.rating_mode == omgwords_pb2.RATED):
                                print(f"[🔥 INCOMING SEEK] Accepting challenge from @{user}...", flush=True)
                                sa = omgseeks_pb2.SoughtGameProcessEvent()
                                sa.request_id = gr.request_id
                                await ws.send(encode_socket_msg(ipc_pb2.SOUGHT_GAME_PROCESS_EVENT, sa))

                    elif msg_type == ipc_pb2.ERROR_MESSAGE:
                        err = errors_pb2.ErrorMessage()
                        err.ParseFromString(payload)
                        print(f"[-] SERVER ERROR: {err.message}", flush=True)

            # Outside lobby socket: Play the game in the dedicated game realm socket!
            if matched_game_id:
                played_ok = await self.play_game(matched_game_id)
                if not played_ok:
                    print("[*] Game did not complete (e.g. aborted). Retrying seek in 2s...", flush=True)
                    await asyncio.sleep(2.0)

        print(f"[+] All {self.target_games} target tournament game(s) completed successfully!", flush=True)
        self.display_rating_card("FINAL AUDITED TOURNAMENT RATING CARD")

def main():
    parser = argparse.ArgumentParser(description="Tessera Woogles.io Live Tournament Bot Runner")
    parser.add_argument("--username", "-u", type=str, default=DEFAULT_USERNAME,
                        help=f"Woogles account username (default: {DEFAULT_USERNAME})")
    parser.add_argument("--api-key", "-k", type=str, default=DEFAULT_API_KEY,
                        help="Woogles account API key")
    parser.add_argument("--lexicon", "-l", type=str, default="CSW24",
                        choices=["CSW24", "NWL23"],
                        help="Tournament Lexicon (default: CSW24)")
    parser.add_argument("--delib", "-d", type=int, default=25,
                        help="Lattice deliberation steps per turn (default: 25)")
    parser.add_argument("--time-minutes", "-t", type=int, default=15,
                        help="Time control minutes per player (default: 15)")
    parser.add_argument("--unrated", action="store_true",
                        help="Play unrated casual games instead of official rated games")
    parser.add_argument("--games", "-g", type=int, default=1,
                        help="Number of tournament games to play (default: 1)")
    parser.add_argument("--vs-bot", type=str, default="HastyBot",
                        choices=list(BOT_CODES.keys()),
                        help="Directly challenge a certified bot (default: HastyBot). Use --lobby for human matching.")
    parser.add_argument("--lobby", action="store_true",
                        help="Post open seek and match humans/bots in the open lobby instead of direct bot challenge")
    parser.add_argument("--test", action="store_true",
                        help="Run engine offline self-test and exit")

    args = parser.parse_args()

    if args.test:
        print("[+] Running Tessera offline self-test...", flush=True)
        eng = TesseraEngine()
        board = [[' ']*15 for _ in range(15)]
        blanks = [[False]*15 for _ in range(15)]
        start = time.time()
        m = eng.compute_move(board, blanks, "RETINAS", 86, 0, delib_steps=args.delib)
        ms = (time.time() - start) * 1000.0
        print(f"    Move generated: {m['word']} at {m['coord']} (+{m['score']} pts in {ms:.1f}ms)", flush=True)
        print("[+] Engine self-test PASSED!", flush=True)
        return

    bot = WooglesBot(
        username=args.username,
        api_key=args.api_key,
        lexicon=args.lexicon,
        delib_steps=args.delib,
        time_minutes=args.time_minutes,
        rated=not args.unrated,
        target_games=args.games,
        vs_bot=args.vs_bot,
        use_lobby=args.lobby
    )

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n[*] Bot runner stopped cleanly.", flush=True)

if __name__ == "__main__":
    main()
