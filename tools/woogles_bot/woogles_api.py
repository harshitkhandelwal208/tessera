"""
Woogles.io API Client
Handles ConnectRPC endpoints, user profile querying, game state streaming, and move execution.
"""

import json
import time
import requests
from typing import Dict, List, Optional, Tuple, Any

class WooglesAPIError(Exception):
    pass

class WooglesClient:
    def __init__(self, api_key: str, username: str, base_url: str = "https://woogles.io/api"):
        self.api_key = api_key.strip()
        self.username = username.strip()
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "User-Agent": f"TesseraScrabbleEngine/1.0 ({self.username})"
        })

    def _call_rpc(self, service_method: str, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        """
        Executes a ConnectRPC HTTP POST call to woogles.io.
        Endpoint pattern: https://woogles.io/api/{service_name}/{method_name}
        """
        url = f"{self.base_url}/{service_method}"
        try:
            resp = self.session.post(url, json=payload, timeout=timeout)
            if resp.status_code == 401 or resp.status_code == 403:
                raise WooglesAPIError(f"Authentication failed ({resp.status_code}): Invalid or missing X-Api-Key for user {self.username}.")
            if resp.status_code != 200:
                raise WooglesAPIError(f"RPC {service_method} failed with status {resp.status_code}: {resp.text}")
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise WooglesAPIError(f"Network error calling {url}: {e}")

    def get_user_profile(self, target_user: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetches official user profile including Glicko-2 ratings, games played, and stats.
        """
        user = target_user or self.username
        # Method: user_service.UserService/GetUserProfile
        return self._call_rpc("user_service.UserService/GetUserProfile", {"username": user})

    def get_user_rating(self, target_user: Optional[str] = None, lexicon: str = "CSW24") -> Dict[str, float]:
        """
        Retrieves official Glicko-2 rating, rating deviation (RD), and game count.
        """
        profile = self.get_user_profile(target_user)
        # Profile contains rating cards by lexicon
        ratings = profile.get("ratings", {})
        lex_rating = ratings.get(lexicon) or ratings.get("CSW21") or ratings.get("default", {})
        return {
            "rating": float(lex_rating.get("rating", 1500.0)),
            "rd": float(lex_rating.get("rd", 350.0)),
            "games_played": int(lex_rating.get("games_played", 0)),
            "lexicon": lexicon
        }

    def get_game_metadata(self, game_id: str) -> Dict[str, Any]:
        """
        Retrieves live game metadata for a match.
        """
        return self._call_rpc("game_service.GameMetadataService/GetMetadata", {"game_id": game_id})

    def get_gcg(self, game_id: str) -> str:
        """
        Fetches official GCG (Generic Crossword Game) record for game.
        """
        res = self._call_rpc("game_service.GameMetadataService/GetGCG", {"game_id": game_id})
        return res.get("gcg", "")

    def get_recent_games(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Queries recent games played by this account.
        """
        res = self._call_rpc("game_service.GameMetadataService/GetRecentGames", {
            "username": self.username,
            "num_games": count
        })
        return res.get("games", [])

    def create_match_seek(self, lexicon: str = "CSW24", time_minutes: int = 25, rated: bool = True) -> Dict[str, Any]:
        """
        Posts an open game challenge or lobby seek for other players or bots to accept.
        """
        payload = {
            "challenger": self.username,
            "lexicon": lexicon,
            "time_control": {
                "initial_time_seconds": time_minutes * 60,
                "increment_seconds": 0
            },
            "rated": rated,
            "challenge_rule": "FIVE_POINT" if "CSW" in lexicon else "DOUBLE"
        }
        return self._call_rpc("game_service.GameService/CreateChallenge", payload)

    def submit_move(self, game_id: str, move_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submits an evaluated move to Woogles.
        """
        move_type = move_data.get("type")
        payload: Dict[str, Any] = {
            "game_id": game_id,
            "player": self.username,
        }

        if move_type == "PLAY":
            payload.update({
                "action": "PLAY",
                "coord": move_data["coord"],
                "word": move_data["word"],
                "score": move_data.get("score", 0)
            })
        elif move_type == "EXCHANGE":
            payload.update({
                "action": "EXCHANGE",
                "tiles": move_data.get("tiles", "")
            })
        elif move_type == "PASS":
            payload.update({
                "action": "PASS"
            })

        return self._call_rpc("game_service.GameService/PlayMove", payload)
