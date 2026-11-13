"""
Tessera Engine Python Bridge for Woogles.io
Interfaces directly with libtessera.so via ctypes for high-throughput in-memory evaluation.
"""

import ctypes
import os
from typing import Dict, List, Optional, Tuple

class BridgePlaced(ctypes.Structure):
    _fields_ = [
        ('row', ctypes.c_int),
        ('col', ctypes.c_int),
        ('letter', ctypes.c_char),
        ('is_blank', ctypes.c_bool),
    ]

class BridgeMove(ctypes.Structure):
    _fields_ = [
        ('type', ctypes.c_int),       # 0 = Play, 1 = Exchange, 2 = Pass
        ('row', ctypes.c_int),
        ('col', ctypes.c_int),
        ('horizontal', ctypes.c_bool),
        ('word', ctypes.c_char * 16),
        ('num_placed', ctypes.c_int),
        ('placed', BridgePlaced * 7),
        ('score', ctypes.c_int),
    ]

class TesseraEngine:
    def __init__(self, lib_path: Optional[str] = None, dawg_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if lib_path is None:
            lib_path = os.path.join(base_dir, "libtessera.so")
        if dawg_path is None:
            dawg_path = os.path.join(base_dir, "data", "dawg.bin")

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Tessera shared library not found at: {lib_path}. Run 'make lib' first.")
        if not os.path.exists(dawg_path):
            raise FileNotFoundError(f"Tessera DAWG lexicon not found at: {dawg_path}.")

        self.lib = ctypes.CDLL(lib_path)

        # Set up prototypes
        self.lib.tessera_create_engine.argtypes = [ctypes.c_char_p]
        self.lib.tessera_create_engine.restype = ctypes.c_void_p

        self.lib.tessera_destroy_engine.argtypes = [ctypes.c_void_p]
        self.lib.tessera_destroy_engine.restype = None

        self.lib.tessera_create_tracker.argtypes = []
        self.lib.tessera_create_tracker.restype = ctypes.c_void_p

        self.lib.tessera_destroy_tracker.argtypes = [ctypes.c_void_p]
        self.lib.tessera_destroy_tracker.restype = None

        self.lib.tessera_compute_move.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char * 225,
            ctypes.c_bool * 225,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p
        ]
        self.lib.tessera_compute_move.restype = BridgeMove

        self.engine = self.lib.tessera_create_engine(dawg_path.encode('utf-8'))
        if not self.engine:
            raise RuntimeError(f"Failed to initialize Tessera DAWG engine from {dawg_path}")
        self.tracker = self.lib.tessera_create_tracker()

    def __del__(self):
        if hasattr(self, 'lib') and hasattr(self, 'engine') and self.engine:
            self.lib.tessera_destroy_engine(self.engine)
            self.engine = None
        if hasattr(self, 'lib') and hasattr(self, 'tracker') and self.tracker:
            self.lib.tessera_destroy_tracker(self.tracker)
            self.tracker = None

    def reset_tracker(self):
        """Resets Bayesian imperfect-information opponent tracker for a new game."""
        if self.tracker:
            self.lib.tessera_destroy_tracker(self.tracker)
        self.tracker = self.lib.tessera_create_tracker()

    def compute_move(self,
                     board_grid: List[List[str]],
                     blanks_grid: List[List[bool]],
                     rack: str,
                     bag_count: int,
                     score_diff: int,
                     delib_steps: int = 35) -> Dict:
        """
        Computes optimal move given board grid, rack, bag, and score diff.
        board_grid: 15x15 list of uppercase letters or empty string/space for empty squares.
        blanks_grid: 15x15 list of booleans indicating if cell is a designated blank.
        """
        c_board = (ctypes.c_char * 225)()
        c_blanks = (ctypes.c_bool * 225)()

        for r in range(15):
            for c in range(15):
                idx = r * 15 + c
                ch = board_grid[r][c]
                if ch and ch != ' ' and ch != '.':
                    c_board[idx] = ch.upper().encode('ascii')
                    c_blanks[idx] = bool(blanks_grid[r][c])
                else:
                    c_board[idx] = 0
                    c_blanks[idx] = False

        c_rack = rack.upper().encode('ascii')
        bm = self.lib.tessera_compute_move(
            self.engine,
            c_board,
            c_blanks,
            c_rack,
            int(bag_count),
            int(score_diff),
            int(delib_steps),
            self.tracker
        )

        res = {
            "type": ["PLAY", "EXCHANGE", "PASS"][bm.type],
            "score": bm.score
        }

        if bm.type == 0: # PLAY
            row = bm.row
            col = bm.col
            word = bm.word.decode('ascii')
            horiz = bm.horizontal
            num_placed = bm.num_placed
            placed_tiles = []
            for p in range(num_placed):
                pt = bm.placed[p]
                placed_tiles.append({
                    "row": pt.row,
                    "col": pt.col,
                    "letter": pt.letter.decode('ascii'),
                    "is_blank": pt.is_blank
                })

            # Woogles coordinate format:
            # Horizontal play on row 8 starting at column D: "8D"
            # Vertical play on col D starting at row 8: "D8"
            col_letter = chr(ord('A') + col)
            row_num = row + 1

            if horiz:
                coord = f"{row_num}{col_letter}"
            else:
                coord = f"{col_letter}{row_num}"

            # Word with lowercase for blanks
            formatted_word = []
            for k in range(len(word)):
                cur_r = row if horiz else row + k
                cur_c = col + k if horiz else col
                is_blank = False
                for pt in placed_tiles:
                    if pt["row"] == cur_r and pt["col"] == cur_c:
                        is_blank = pt["is_blank"]
                        break
                formatted_word.append(word[k].lower() if is_blank else word[k].upper())

            res.update({
                "row": row,
                "col": col,
                "horizontal": horiz,
                "coord": coord,
                "word": "".join(formatted_word),
                "num_placed": num_placed,
                "placed": placed_tiles
            })

        elif bm.type == 1: # EXCHANGE
            tiles = []
            for p in range(bm.num_placed):
                pt = bm.placed[p]
                tiles.append('?' if pt.is_blank else pt.letter.decode('ascii'))
            res["tiles"] = "".join(tiles)

        return res
