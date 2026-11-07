/**
 * TESSERA — Scrabble Engine Web UI Controller
 * Championship Two-Player Analysis Arena & Demonstration Station
 * Communicates with the native C11 engine via JSON REST API.
 */

const BOARD_SIZE = 15;
const TILE_SCORES = {
    'A':1, 'B':3, 'C':3, 'D':2, 'E':1, 'F':4, 'G':2, 'H':4, 'I':1, 'J':8,
    'K':5, 'L':1, 'M':3, 'N':1, 'O':1, 'P':3, 'Q':10, 'R':1, 'S':1, 'T':1,
    'U':1, 'V':4, 'W':4, 'X':8, 'Y':4, 'Z':10, '?':0
};

const TOTAL_DISTRIBUTION = {
    'A':9, 'B':2, 'C':2, 'D':4, 'E':12, 'F':2, 'G':3, 'H':2, 'I':9, 'J':1,
    'K':1, 'L':4, 'M':2, 'N':6, 'O':8, 'P':2, 'Q':1, 'R':6, 'S':4, 'T':6,
    'U':4, 'V':2, 'W':2, 'X':1, 'Y':2, 'Z':1, '?':2
};

const OPPONENT_PROFILES = {
    "6": {
        name: "MAGPIE Champion",
        elo: "(2150 Elo)",
        desc: "2-Ply Realization Simulation against Bayesian tile refills",
        rackTitle: "Opponent Rack — MAGPIE (2150 Elo)"
    },
    "5": {
        name: "Quackle Simulation",
        elo: "(2100 Elo)",
        desc: "Monte Carlo Playout Rollouts with equity evaluation",
        rackTitle: "Opponent Rack — Quackle (2100 Elo)"
    },
    "4": {
        name: "Neural Static",
        elo: "(1950 Elo)",
        desc: "448-dim Position-Keyed NNUE 1-Ply Static Evaluator",
        rackTitle: "Opponent Rack — Neural Static (1950 Elo)"
    },
    "3": {
        name: "Heuristic Grandmaster",
        elo: "(1850 Elo)",
        desc: "8-Phase Leave Evaluation with Vowel-Consonant Balance",
        rackTitle: "Opponent Rack — Heuristic GM (1850 Elo)"
    },
    "2": {
        name: "Bingo Hunter",
        elo: "(1750 Elo)",
        desc: "Prioritizes TISANE/RETINAS stem leave retention",
        rackTitle: "Opponent Rack — Bingo Hunter (1750 Elo)"
    },
    "1": {
        name: "Defensive Turtler",
        elo: "(1650 Elo)",
        desc: "Tight board control, minimizes triple word openings",
        rackTitle: "Opponent Rack — Turtler (1650 Elo)"
    },
    "0": {
        name: "Baseline Greedy",
        elo: "(1550 Elo)",
        desc: "1-Ply Pure Score Maximizer (no leave synergy)",
        rackTitle: "Opponent Rack — Baseline Greedy (1550 Elo)"
    },
    "human": {
        name: "Human Challenger",
        elo: "(Interactive)",
        desc: "Manual play mode — Place tiles and validate moves",
        rackTitle: "Your Rack (Interactive Challenger)"
    }
};

// Global State
let gameState = null;
let pendingPlaced = []; // Array of {row, col, letter, is_blank, original_rack_index}
let selectedRackIndex = -1;
let moveHistory = [];
let isAiVsAiRunning = false;
let aiVsAiTimer = null;
let selectedOpponentKey = "6";
let lastMoveCoords = [];

// Cached DOM Elements
const boardEl = document.getElementById("scrabble-board");
const colLabelsEl = document.getElementById("col-labels");
const rowLabelsEl = document.getElementById("row-labels");
const rackEl = document.getElementById("player-rack");
const oppRackEl = document.getElementById("opp-rack-tiles");
const statusMsgEl = document.getElementById("status-message");
const scorePlayerEl = document.getElementById("score-player");
const scoreAiEl = document.getElementById("score-ai");
const bagCountEl = document.getElementById("bag-count");
const hintsTbodyEl = document.getElementById("hints-tbody");
const historyTbodyEl = document.getElementById("history-tbody");
const tileTrackerGridEl = document.getElementById("tile-tracker-grid");
const leadPillEl = document.getElementById("match-lead-pill");
const oppSelectEl = document.getElementById("opponent-select");

// Initialize Board UI
function initBoardGrid() {
    colLabelsEl.innerHTML = "";
    rowLabelsEl.innerHTML = "";
    boardEl.innerHTML = "";

    // Columns A-O
    for (let c = 0; c < BOARD_SIZE; c++) {
        const colLabel = document.createElement("div");
        colLabel.textContent = String.fromCharCode(65 + c);
        colLabelsEl.appendChild(colLabel);
    }

    // Rows 1-15
    for (let r = 0; r < BOARD_SIZE; r++) {
        const rowLabel = document.createElement("div");
        rowLabel.textContent = (r + 1).toString();
        rowLabelsEl.appendChild(rowLabel);
    }

    // 15x15 Cells
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const cell = document.createElement("div");
            cell.className = "cell";
            cell.dataset.row = r;
            cell.dataset.col = c;

            // Multiplier styling
            const mult = getCellMultiplier(r, c);
            if (mult === "TW") { cell.classList.add("cell-tw"); cell.textContent = "3W"; }
            else if (mult === "DW") { cell.classList.add("cell-dw"); cell.textContent = "2W"; }
            else if (mult === "TL") { cell.classList.add("cell-tl"); cell.textContent = "3L"; }
            else if (mult === "DL") { cell.classList.add("cell-dl"); cell.textContent = "2L"; }
            else if (mult === "STAR") { cell.classList.add("cell-star"); cell.textContent = "★"; }

            cell.addEventListener("click", () => handleCellClick(r, c));
            boardEl.appendChild(cell);
        }
    }
}

function getCellMultiplier(r, c) {
    if ((r === 0 || r === 7 || r === 14) && (c === 0 || c === 7 || c === 14)) {
        if (r === 7 && c === 7) return "STAR";
        return "TW";
    }
    if (r === c || r === (14 - c)) {
        if ((r >= 1 && r <= 4) || (r >= 10 && r <= 13) || r === 7) return "DW";
    }
    if ((r === 1 || r === 13) && (c === 5 || c === 9)) return "TL";
    if ((r === 5 || r === 9) && (c === 1 || c === 5 || c === 9 || c === 13)) return "TL";

    if ((r === 0 || r === 14) && (c === 3 || c === 11)) return "DL";
    if ((r === 2 || r === 12) && (c === 6 || c === 8)) return "DL";
    if ((r === 3 || r === 11) && (c === 0 || c === 7 || c === 14)) return "DL";
    if ((r === 6 || r === 8) && (c === 2 || c === 6 || c === 8 || c === 12)) return "DL";
    if (r === 7 && (c === 3 || c === 11)) return "DL";

    return "";
}

function createTileElement(letter, isBlank = false, isPending = false) {
    const tile = document.createElement("div");
    tile.className = "tile" + (isBlank ? " blank-tile" : "") + (isPending ? " pending" : "");

    const lSpan = document.createElement("span");
    lSpan.className = "letter";
    lSpan.textContent = isBlank ? (letter || "?").toLowerCase() : letter;
    tile.appendChild(lSpan);

    const sub = document.createElement("span");
    sub.className = "sub";
    sub.textContent = isBlank ? 0 : (TILE_SCORES[letter] || 0);
    tile.appendChild(sub);

    return tile;
}

function updateOpponentHeader() {
    const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];
    const elName = document.getElementById("opp-display-name");
    const elElo = document.getElementById("opp-display-elo");
    const elDesc = document.getElementById("opp-display-desc");
    const elTitle = document.getElementById("opp-rack-title");
    const humanTools = document.getElementById("human-tools");
    const humanActions = document.getElementById("human-actions");
    const playerTitle = document.getElementById("player-rack-title");

    if (elName) elName.textContent = prof.name;
    if (elElo) elElo.textContent = prof.elo;
    if (elDesc) elDesc.textContent = prof.desc;
    if (elTitle) elTitle.textContent = prof.rackTitle;

    if (selectedOpponentKey === "human") {
        if (humanTools) humanTools.style.display = "flex";
        if (humanActions) humanActions.style.display = "flex";
        if (playerTitle) playerTitle.textContent = "Your Rack (Interactive Challenger)";
    } else {
        if (humanTools) humanTools.style.display = "none";
        if (humanActions) humanActions.style.display = "none";
        if (playerTitle) playerTitle.textContent = "Tessera AI Rack (2016 TPR)";
    }
}

// Render Board & Racks from State
function renderState(state) {
    if (!state) return;
    gameState = state;

    scorePlayerEl.textContent = state.player_score;
    scoreAiEl.textContent = state.ai_score;
    bagCountEl.textContent = state.bag_count;

    // Render Lead Pill
    const pScore = state.player_score;
    const aScore = state.ai_score;
    const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];
    if (leadPillEl) {
        if (pScore === aScore) {
            leadPillEl.textContent = `Tied ${pScore} \u2013 ${aScore}`;
        } else if (aScore > pScore) {
            leadPillEl.textContent = `Tessera +${aScore - pScore} (${aScore}\u2013${pScore})`;
        } else {
            const oppShort = prof.name.split(" ")[0];
            leadPillEl.textContent = `${oppShort} +${pScore - aScore} (${pScore}\u2013${aScore})`;
        }
    }

    // Active Turn Indicators
    const oppDot = document.getElementById("opp-turn-dot");
    const tesseraDot = document.getElementById("tessera-turn-dot");
    const oppBox = document.getElementById("player-box-opp");
    const tesseraBox = document.getElementById("player-box-tessera");
    const oppBadge = document.getElementById("opp-status-badge");
    const tesseraBadge = document.getElementById("tessera-status-badge");

    if (state.turn === 0) {
        // Opponent's turn
        if (oppDot) oppDot.classList.add("active");
        if (oppBox) oppBox.classList.add("active-turn-box");
        if (oppBadge) {
            oppBadge.textContent = "Active Turn";
            oppBadge.classList.add("active-turn");
        }

        if (tesseraDot) tesseraDot.classList.remove("active");
        if (tesseraBox) tesseraBox.classList.remove("active-turn-box");
        if (tesseraBadge) {
            tesseraBadge.textContent = "Waiting";
            tesseraBadge.classList.remove("active-turn");
        }
    } else {
        // Tessera's turn
        if (tesseraDot) tesseraDot.classList.add("active");
        if (tesseraBox) tesseraBox.classList.add("active-turn-box");
        if (tesseraBadge) {
            tesseraBadge.textContent = "Active Turn";
            tesseraBadge.classList.add("active-turn");
        }

        if (oppDot) oppDot.classList.remove("active");
        if (oppBox) oppBox.classList.remove("active-turn-box");
        if (oppBadge) {
            oppBadge.textContent = "Waiting";
            oppBadge.classList.remove("active-turn");
        }
    }

    // Render Board Cells & Highlights
    const cells = boardEl.querySelectorAll(".cell");
    cells.forEach(cell => {
        const r = parseInt(cell.dataset.row);
        const c = parseInt(cell.dataset.col);

        cell.classList.remove("last-move-cell");
        const oldTile = cell.querySelector(".tile");
        if (oldTile) oldTile.remove();

        // Highlight cells from the latest move
        if (lastMoveCoords.some(([lr, lc]) => lr === r && lc === c)) {
            cell.classList.add("last-move-cell");
        }

        // Committed board tile
        const boardCell = state.board[r][c];
        if (boardCell.tile) {
            const tileEl = createTileElement(boardCell.tile, boardCell.is_blank);
            cell.appendChild(tileEl);
        } else {
            // Pending human placement
            const pending = pendingPlaced.find(p => p.row === r && p.col === c);
            if (pending) {
                const tileEl = createTileElement(pending.letter, pending.is_blank, true);
                cell.appendChild(tileEl);
            }
        }
    });

    renderOpponentRack();
    renderPlayerRack();
    renderTileTracker();

    // Telemetry Phase & Openness
    const elPhase = document.getElementById("telemetry-phase-bucket");
    const elOpen = document.getElementById("telemetry-openness");
    if (elPhase) {
        const bc = state.bag_count;
        if (bc >= 80) elPhase.textContent = `Opening (80-100)`;
        else if (bc >= 60) elPhase.textContent = `Early Game (60-79)`;
        else if (bc >= 40) elPhase.textContent = `Midgame (40-59)`;
        else if (bc >= 20) elPhase.textContent = `Late Midgame (20-39)`;
        else if (bc >= 10) elPhase.textContent = `Pre-Endgame (10-19)`;
        else if (bc >= 1)  elPhase.textContent = `Bag Exhaustion (1-9)`;
        else elPhase.textContent = `Exact Endgame Solver (0 tiles)`;
    }
    if (elOpen) {
        let occupied = 0;
        let twExposed = 0;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (state.board[r][c].tile) occupied++;
                else if (state.board[r][c].mult === 4) { // MULT_TW
                    const hasAdj = (r > 0 && state.board[r-1][c].tile) ||
                                   (r < 14 && state.board[r+1][c].tile) ||
                                   (c > 0 && state.board[r][c-1].tile) ||
                                   (c < 14 && state.board[r][c+1].tile);
                    if (hasAdj) twExposed++;
                }
            }
        }
        if (occupied < 8) elOpen.textContent = "Closed Center";
        else if (twExposed >= 3) elOpen.textContent = "Wide-Open Perimeter";
        else if (twExposed >= 1) elOpen.textContent = "Hotspots Exposed";
        else elOpen.textContent = "Standard Openness";
    }
}

function renderOpponentRack() {
    if (!oppRackEl || !gameState) return;
    oppRackEl.innerHTML = "";

    const rackStr = (selectedOpponentKey === "human") ? gameState.ai_rack : gameState.player_rack;
    if (!rackStr) return;

    for (let i = 0; i < rackStr.length; i++) {
        const ch = rackStr[i];
        const isBlank = (ch === '?');
        const tileEl = createTileElement(ch, isBlank);
        oppRackEl.appendChild(tileEl);
    }
}

function renderPlayerRack() {
    rackEl.innerHTML = "";
    if (!gameState) return;

    const isHuman = (selectedOpponentKey === "human");
    const rackStr = isHuman ? gameState.player_rack : gameState.ai_rack;
    if (!rackStr) return;

    if (!isHuman) {
        // Read-only Tessera AI rack display in AI-vs-AI mode
        for (let i = 0; i < rackStr.length; i++) {
            const ch = rackStr[i];
            const tileEl = createTileElement(ch, ch === '?');
            rackEl.appendChild(tileEl);
        }
        return;
    }

    // Interactive Human Rack
    const availableTiles = [];
    const usedIndices = new Set(pendingPlaced.map(p => p.original_rack_index));

    for (let i = 0; i < rackStr.length; i++) {
        if (!usedIndices.has(i)) {
            availableTiles.push({ letter: rackStr[i], index: i });
        }
    }

    availableTiles.forEach(t => {
        const isBlank = (t.letter === '?');
        const tileEl = createTileElement(t.letter, isBlank);
        if (selectedRackIndex === t.index) {
            tileEl.classList.add("selected");
        }

        tileEl.addEventListener("click", () => {
            if (selectedRackIndex === t.index) {
                selectedRackIndex = -1;
            } else {
                selectedRackIndex = t.index;
            }
            renderPlayerRack();
        });

        rackEl.appendChild(tileEl);
    });
}

function handleCellClick(r, c) {
    if (isAiVsAiRunning || selectedOpponentKey !== "human") return;

    // 1. If square already has a pending placed tile, return it to rack
    const pendingIdx = pendingPlaced.findIndex(p => p.row === r && p.col === c);
    if (pendingIdx >= 0) {
        pendingPlaced.splice(pendingIdx, 1);
        renderState(gameState);
        return;
    }

    // 2. If square already has a permanently committed tile, ignore
    if (gameState.board[r][c].tile) {
        return;
    }

    // 3. If a rack tile is currently selected, place it here
    if (selectedRackIndex >= 0) {
        const letter = gameState.player_rack[selectedRackIndex];
        let placedLetter = letter;
        let isBlank = false;

        if (letter === '?') {
            const chosen = prompt("Enter letter for blank tile (A-Z):", "A");
            if (!chosen || !/^[a-zA-Z]$/.test(chosen)) return;
            placedLetter = chosen.toUpperCase();
            isBlank = true;
        }

        pendingPlaced.push({
            row: r,
            col: c,
            letter: placedLetter,
            is_blank: isBlank,
            original_rack_index: selectedRackIndex
        });

        selectedRackIndex = -1;
        renderState(gameState);
    }
}

// Unseen Bag Tracker (Tab 4)
function renderTileTracker() {
    if (!tileTrackerGridEl || !gameState) return;
    tileTrackerGridEl.innerHTML = "";

    const counts = { ...TOTAL_DISTRIBUTION };

    // Subtract board tiles
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            const cell = gameState.board[r][c];
            if (cell.tile) {
                if (cell.is_blank) counts['?']--;
                else counts[cell.tile]--;
            }
        }
    }

    // Subtract known player rack
    const activeRack = (selectedOpponentKey === "human") ? gameState.player_rack : "";
    for (const ch of activeRack) {
        counts[ch]--;
    }

    const letters = Object.keys(counts).sort();
    for (const ch of letters) {
        const item = document.createElement("div");
        item.className = "tracker-item" + (counts[ch] <= 0 ? " exhausted" : "");

        const l = document.createElement("span");
        l.className = "tracker-letter";
        l.textContent = (ch === '?') ? "BLANK" : ch;

        const cnt = document.createElement("span");
        cnt.className = "tracker-count";
        cnt.textContent = Math.max(0, counts[ch]);

        item.appendChild(l);
        item.appendChild(cnt);
        tileTrackerGridEl.appendChild(item);
    }
}

// Highlight Newly Placed Move on Board
function highlightMoveOnBoard(move) {
    lastMoveCoords = [];
    if (!move || move.type !== 0 || !move.word) return;
    const len = move.word.length;
    for (let i = 0; i < len; i++) {
        const r = (move.dir === "V") ? move.row + i : move.row;
        const c = (move.dir === "H") ? move.col + i : move.col;
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            lastMoveCoords.push([r, c]);
        }
    }
}

function updateTelemetryMetrics(m) {
    if (!m) return;
    const elNodes = document.getElementById("telemetry-nodes");
    const elTrans = document.getElementById("telemetry-transpositions");
    const elPasses = document.getElementById("telemetry-dirty-passes");
    const elConf = document.getElementById("telemetry-court-conf");
    const elPalimp = document.getElementById("telemetry-palimpsest");
    if (elNodes) elNodes.textContent = (m.nodes || 0).toLocaleString();
    if (elTrans) elTrans.textContent = (m.transpositions || 0).toLocaleString();
    if (elPasses) elPasses.textContent = (m.settle_passes || 0).toLocaleString();
    if (elConf) elConf.textContent = ((m.confidence || 0) * 100).toFixed(1) + "%";
    if (elPalimp) elPalimp.textContent = (m.palimpsest_bias || 0).toFixed(2) + " pts";
}

function addHistoryRow(player, word, pos, score, total) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
        <td>${moveHistory.length + 1}</td>
        <td><strong>${player}</strong></td>
        <td><code>${word}</code></td>
        <td>${pos}</td>
        <td>+${score}</td>
        <td><strong>${total}</strong></td>
    `;
    if (historyTbodyEl.querySelector(".placeholder-text")) {
        historyTbodyEl.innerHTML = "";
    }
    historyTbodyEl.prepend(tr);
    moveHistory.push({ player, word, pos, score, total });
}

function declareGameOver(state) {
    const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];
    let winner = "Tie Game!";
    if (state.ai_score > state.player_score) {
        winner = `Tessera AI wins by ${state.ai_score - state.player_score} points!`;
    } else if (state.player_score > state.ai_score) {
        winner = `${prof.name} wins by ${state.player_score - state.ai_score} points!`;
    }
    statusMsgEl.innerHTML = `<strong style="color: #f59e0b;">🏁 MATCH FINISHED:</strong> ${winner} Final Score: <strong>${state.player_score}</strong> (${prof.name}) &ndash; <strong>${state.ai_score}</strong> (Tessera).`;
}

// API Communication
async function apiGetState() {
    const res = await fetch("/api/state");
    return await res.json();
}

async function apiNewGame() {
    const res = await fetch("/api/new_game");
    return await res.json();
}

async function apiPlayMove(row, col, dir, word) {
    const res = await fetch("/api/play_move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ row, col, dir, word })
    });
    return await res.json();
}

async function apiAiMove() {
    const res = await fetch("/api/ai_move");
    return await res.json();
}

async function apiOpponentMove(botType = "6") {
    const res = await fetch(`/api/opponent_move?bot_type=${encodeURIComponent(botType)}`);
    return await res.json();
}

async function apiGetHint() {
    const res = await fetch("/api/hint");
    return await res.json();
}

async function apiPass() {
    const res = await fetch("/api/pass", { method: "POST" });
    return await res.json();
}

async function apiExchange(tiles) {
    const res = await fetch("/api/exchange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tiles })
    });
    return await res.json();
}

async function apiCheckWord(word) {
    const res = await fetch(`/api/check_word?word=${encodeURIComponent(word)}`);
    return await res.json();
}

// Centralized Move Result Processor
async function handleMoveResult(res, actorName, actorRole) {
    if (res.status !== "ok") {
        if (statusMsgEl) statusMsgEl.textContent = `${actorName} error: ${res.message || "Move failed"}`;
        return;
    }

    if (res.move.type === 0) { // MOVE_PLAY
        const pos = `${String.fromCharCode(65 + res.move.col)}${res.move.row + 1} (${res.move.dir})`;
        const total = (actorRole === "Opponent") ? res.player_score : res.ai_score;
        addHistoryRow(actorName, res.move.word, pos, res.move.score, total);
        highlightMoveOnBoard(res.move);

        const nextActor = (actorRole === "Opponent") ? "Tessera AI (2016 TPR)" : actorName;
        if (statusMsgEl) statusMsgEl.innerHTML = `<strong>${actorName}</strong> played <code>${res.move.word}</code> for <strong>+${res.move.score} pts</strong> (${res.time_ms.toFixed(1)} ms). Next: <em>${nextActor}</em>`;
    } else if (res.move.type === 1) {
        const total = (actorRole === "Opponent") ? res.player_score : res.ai_score;
        addHistoryRow(actorName, "(Exchanged Tiles)", "-", 0, total);
        if (statusMsgEl) statusMsgEl.innerHTML = `<strong>${actorName}</strong> exchanged tiles. Next turn!`;
    } else {
        const total = (actorRole === "Opponent") ? res.player_score : res.ai_score;
        addHistoryRow(actorName, "(Pass)", "-", 0, total);
        if (statusMsgEl) statusMsgEl.innerHTML = `<strong>${actorName}</strong> passed. Next turn!`;
    }

    if (res.metrics) {
        updateTelemetryMetrics(res.metrics);
    }
    if (res.phase) {
        const elPhase = document.getElementById("telemetry-phase-bucket");
        if (elPhase) elPhase.textContent = res.phase;
    }
    if (res.openness) {
        const elOpen = document.getElementById("telemetry-openness");
        if (elOpen) elOpen.textContent = res.openness;
    }
    if (res.endgame) {
        if (statusMsgEl) statusMsgEl.innerHTML += ` &bull; <span style="color:#2ecc71;font-weight:bold;">👑 EXACT ENDGAME MINIMAX ACTIVE</span>`;
    }

    // Refresh state from engine
    const fresh = await apiGetState();
    renderState(fresh);
    if (fresh.game_over) {
        declareGameOver(fresh);
    }
}

// Step 1 Single Turn (For live teacher demonstration & walkthrough)
async function stepSingleTurn() {
    if (isAiVsAiRunning) {
        toggleAiVsAi(); // Pause continuous loop if active
    }
    if (!gameState || gameState.game_over) {
        if (statusMsgEl) statusMsgEl.textContent = "Match is over. Click 'Reset Match' to start a new match.";
        return;
    }

    const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];

    if (gameState.turn === 0) {
        // Opponent's turn
        if (selectedOpponentKey === "human") {
            if (statusMsgEl) statusMsgEl.textContent = "Human turn: Place your tiles and click 'Play Move' or 'Suggest Best Move'.";
            return;
        }

        if (statusMsgEl) statusMsgEl.innerHTML = `<span class="spinner"></span> <strong>${prof.name}</strong> is calculating move...`;
        const res = await apiOpponentMove(selectedOpponentKey);
        await handleMoveResult(res, prof.name, "Opponent");
    } else {
        // Tessera's turn
        if (statusMsgEl) statusMsgEl.innerHTML = `<span class="spinner"></span> <strong>Tessera AI (2016 TPR)</strong> is searching Belief Lattice DAG...`;
        const res = await apiAiMove();
        await handleMoveResult(res, "Tessera AI", "Tessera");
    }
}

// Continuous AI vs AI Demonstration Mode
async function toggleAiVsAi() {
    const btn = document.getElementById("btn-ai-vs-ai");
    if (isAiVsAiRunning) {
        isAiVsAiRunning = false;
        clearTimeout(aiVsAiTimer);
        if (btn) {
            btn.textContent = "▶ Start AI Match";
            btn.classList.remove("btn-accent");
            btn.classList.add("btn-primary");
        }
        if (statusMsgEl) statusMsgEl.textContent = "AI match paused. Click 'Step 1 Turn' or '▶ Start AI Match' to continue.";
    } else {
        if (selectedOpponentKey === "human") {
            alert("Please select an AI opponent archetype from the dropdown to run an AI vs AI match.");
            return;
        }
        if (gameState && gameState.game_over) {
            if (statusMsgEl) statusMsgEl.textContent = "Game is already over. Reset match first.";
            return;
        }
        isAiVsAiRunning = true;
        if (btn) {
            btn.textContent = "⏸ Pause AI Match";
            btn.classList.remove("btn-primary");
            btn.classList.add("btn-accent");
        }

        runAiVsAiStep();
    }
}

async function runAiVsAiStep() {
    if (!isAiVsAiRunning) return;
    if (!gameState || gameState.game_over) {
        isAiVsAiRunning = false;
        const btn = document.getElementById("btn-ai-vs-ai");
        if (btn) {
            btn.textContent = "▶ Start AI Match";
            btn.classList.remove("btn-accent");
            btn.classList.add("btn-primary");
        }
        return;
    }

    const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];

    if (gameState.turn === 0) {
        if (statusMsgEl) statusMsgEl.innerHTML = `<span class="spinner"></span> <strong>${prof.name}</strong> is calculating move...`;
        const res = await apiOpponentMove(selectedOpponentKey);
        await handleMoveResult(res, prof.name, "Opponent");
    } else {
        if (statusMsgEl) statusMsgEl.innerHTML = `<span class="spinner"></span> <strong>Tessera AI (2016 TPR)</strong> is searching Belief Lattice DAG...`;
        const res = await apiAiMove();
        await handleMoveResult(res, "Tessera AI", "Tessera");
    }

    if (gameState && gameState.game_over) {
        isAiVsAiRunning = false;
        const btn = document.getElementById("btn-ai-vs-ai");
        if (btn) {
            btn.textContent = "▶ Start AI Match";
            btn.classList.remove("btn-accent");
            btn.classList.add("btn-primary");
        }
        declareGameOver(gameState);
        return;
    }

    if (isAiVsAiRunning) {
        aiVsAiTimer = setTimeout(runAiVsAiStep, 1300); // 1.3s cadence for smooth viewing & explanation
    }
}

// Interactive Human Move Submission
async function submitPlayMove() {
    if (pendingPlaced.length === 0) {
        alert("Please place at least one tile on the board.");
        return;
    }

    const rows = pendingPlaced.map(p => p.row);
    const cols = pendingPlaced.map(p => p.col);
    const sameRow = rows.every(r => r === rows[0]);
    const sameCol = cols.every(c => c === cols[0]);

    if (!sameRow && !sameCol) {
        alert("Invalid placement: all tiles must be placed in a single straight line (row or column).");
        return;
    }

    let dir = sameRow ? "H" : "V";
    if (pendingPlaced.length === 1) {
        const r = pendingPlaced[0].row;
        const c = pendingPlaced[0].col;
        const hasH = (c > 0 && gameState.board[r][c - 1].tile) || (c < 14 && gameState.board[r][c + 1].tile);
        const hasV = (r > 0 && gameState.board[r - 1][c].tile) || (r < 14 && gameState.board[r + 1][c].tile);
        if (hasV && !hasH) dir = "V";
        else dir = "H";
    }

    let startR, startC, wordChars = [];
    if (dir === "H") {
        const r = rows[0];
        let minC = Math.min(...cols);
        let maxC = Math.max(...cols);
        while (minC > 0 && gameState.board[r][minC - 1].tile) minC--;
        while (maxC < 14 && gameState.board[r][maxC + 1].tile) maxC++;

        startR = r;
        startC = minC;

        for (let c = minC; c <= maxC; c++) {
            const p = pendingPlaced.find(x => x.row === r && x.col === c);
            if (p) wordChars.push(p.letter);
            else if (gameState.board[r][c].tile) wordChars.push(gameState.board[r][c].tile);
            else {
                alert("Invalid move: word has empty gaps!");
                return;
            }
        }
    } else {
        const c = cols[0];
        let minR = Math.min(...rows);
        let maxR = Math.max(...rows);
        while (minR > 0 && gameState.board[minR - 1][c].tile) minR--;
        while (maxR < 14 && gameState.board[maxR + 1][c].tile) maxR++;

        startR = minR;
        startC = c;

        for (let r = minR; r <= maxR; r++) {
            const p = pendingPlaced.find(x => x.row === r && x.col === c);
            if (p) wordChars.push(p.letter);
            else if (gameState.board[r][c].tile) wordChars.push(gameState.board[r][c].tile);
            else {
                alert("Invalid move: word has empty gaps!");
                return;
            }
        }
    }

    const fullWord = wordChars.join("");
    statusMsgEl.textContent = `Validating move ${fullWord}...`;

    const res = await apiPlayMove(startR, startC, dir, fullWord);
    if (res.status === "ok") {
        pendingPlaced = [];
        const pos = `${String.fromCharCode(65 + startC)}${startR + 1} (${dir})`;
        addHistoryRow("Player", fullWord, pos, res.score, res.player_score);
        highlightMoveOnBoard({ row: startR, col: startC, dir, word: fullWord, type: 0 });
        statusMsgEl.textContent = `Player scored ${res.score} points with ${fullWord}! AI is thinking...`;

        const freshState = await apiGetState();
        renderState(freshState);

        setTimeout(triggerAiTurn, 600);
    } else {
        alert("Move rejected: " + res.message);
        statusMsgEl.textContent = "Move rejected: " + res.message;
    }
}

async function triggerAiTurn() {
    statusMsgEl.innerHTML = `<span class="spinner"></span> Tessera AI is searching Belief Lattice DAG...`;
    const res = await apiAiMove();
    handleMoveResult(res, "Tessera AI", "Tessera");
}

// Setup Event Listeners
function setupEvents() {
    // Opponent Selector
    if (oppSelectEl) {
        oppSelectEl.addEventListener("change", () => {
            selectedOpponentKey = oppSelectEl.value;
            updateOpponentHeader();
            renderState(gameState);
            const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];
            if (statusMsgEl) statusMsgEl.textContent = `Opponent profile switched to ${prof.name} ${prof.elo}.`;
        });
    }

    // Reset Match
    const btnNew = document.getElementById("btn-new-game");
    if (btnNew) {
        btnNew.addEventListener("click", async () => {
            if (isAiVsAiRunning) {
                toggleAiVsAi();
            }
            pendingPlaced = [];
            lastMoveCoords = [];
            moveHistory = [];
            if (historyTbodyEl) historyTbodyEl.innerHTML = `<tr><td colspan="6" class="placeholder-text">No moves played yet.</td></tr>`;
            const state = await apiNewGame();
            renderState(state);
            const prof = OPPONENT_PROFILES[selectedOpponentKey] || OPPONENT_PROFILES["6"];
            if (statusMsgEl) statusMsgEl.textContent = `New match initialized. ${prof.name} vs Tessera AI. Click 'Step 1 Turn' or '▶ Start AI Match' to begin!`;
        });
    }

    // Step 1 Turn Button
    const btnStep = document.getElementById("btn-step-turn");
    if (btnStep) {
        btnStep.addEventListener("click", stepSingleTurn);
    }

    // AI vs AI Toggle Button
    const btnAi = document.getElementById("btn-ai-vs-ai");
    if (btnAi) {
        btnAi.addEventListener("click", toggleAiVsAi);
    }

    // Human Controls
    const btnRecall = document.getElementById("btn-recall-tiles");
    if (btnRecall) {
        btnRecall.addEventListener("click", () => {
            pendingPlaced = [];
            selectedRackIndex = -1;
            renderState(gameState);
        });
    }

    const btnShuffle = document.getElementById("btn-shuffle-rack");
    if (btnShuffle) {
        btnShuffle.addEventListener("click", () => {
            if (!gameState || !gameState.player_rack) return;
            const arr = gameState.player_rack.split("");
            for (let i = arr.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
            gameState.player_rack = arr.join("");
            renderPlayerRack();
        });
    }

    const btnPlay = document.getElementById("btn-play-move");
    if (btnPlay) {
        btnPlay.addEventListener("click", submitPlayMove);
    }

    const btnAiPlay = document.getElementById("btn-ai-play");
    if (btnAiPlay) {
        btnAiPlay.addEventListener("click", triggerAiTurn);
    }

    const btnPass = document.getElementById("btn-pass");
    if (btnPass) {
        btnPass.addEventListener("click", async () => {
            if (confirm("Are you sure you want to pass your turn?")) {
                await apiPass();
                addHistoryRow("Player", "(Pass)", "-", 0, gameState.player_score);
                if (statusMsgEl) statusMsgEl.textContent = "Player passed. AI is thinking...";
                setTimeout(triggerAiTurn, 500);
            }
        });
    }

    // Candidate Plays Hint
    const btnHint = document.getElementById("btn-get-hint");
    if (btnHint) {
        btnHint.addEventListener("click", async () => {
            const loading = document.getElementById("engine-hint-loading");
            if (loading) loading.classList.remove("hidden");
            const res = await apiGetHint();
            if (loading) loading.classList.add("hidden");

            if (res.status === "ok" && res.hints) {
                if (res.phase) {
                    const elPhase = document.getElementById("telemetry-phase-bucket");
                    if (elPhase) elPhase.textContent = res.phase;
                }
                if (res.openness) {
                    const elOpen = document.getElementById("telemetry-openness");
                    if (elOpen) elOpen.textContent = res.openness;
                }
                if (hintsTbodyEl) {
                    hintsTbodyEl.innerHTML = "";
                    res.hints.forEach((h, idx) => {
                        const tr = document.createElement("tr");
                        if (idx === 0) tr.classList.add("top-candidate");
                        const pos = `${String.fromCharCode(65 + h.col)}${h.row + 1} (${h.dir})`;
                        tr.innerHTML = `
                            <td>#${idx + 1}</td>
                            <td><strong>${h.word}</strong></td>
                            <td>${pos}</td>
                            <td>+${h.score}</td>
                            <td>${h.leave || "\u2014"}</td>
                            <td>${h.equity.toFixed(1)}</td>
                        `;
                        hintsTbodyEl.appendChild(tr);
                    });
                }
                const tabBtn = document.querySelector('[data-tab="tab-hints"]');
                if (tabBtn) tabBtn.click();
            }
        });
    }

    // Exchange Modal
    const modalEx = document.getElementById("modal-exchange");
    const modalRackEl = document.getElementById("modal-exchange-rack");
    const btnExchange = document.getElementById("btn-exchange");
    if (btnExchange && modalEx && modalRackEl) {
        btnExchange.addEventListener("click", () => {
            if (gameState.bag_count < 7) {
                alert("Cannot exchange tiles when bag has fewer than 7 tiles.");
                return;
            }
            modalRackEl.innerHTML = "";
            for (let i = 0; i < gameState.player_rack.length; i++) {
                const ch = gameState.player_rack[i];
                const tile = createTileElement(ch, ch === '?');
                tile.dataset.char = ch;
                tile.addEventListener("click", () => tile.classList.toggle("selected"));
                modalRackEl.appendChild(tile);
            }
            modalEx.classList.remove("hidden");
        });
    }

    const btnExchangeCancel = document.getElementById("btn-exchange-cancel");
    if (btnExchangeCancel && modalEx) {
        btnExchangeCancel.addEventListener("click", () => {
            modalEx.classList.add("hidden");
        });
    }

    const btnExchangeConfirm = document.getElementById("btn-exchange-confirm");
    if (btnExchangeConfirm && modalEx && modalRackEl) {
        btnExchangeConfirm.addEventListener("click", async () => {
            const selectedTiles = modalRackEl.querySelectorAll(".tile.selected");
            if (selectedTiles.length === 0) {
                alert("Please select at least one tile to exchange.");
                return;
            }
            const discard = Array.from(selectedTiles).map(t => t.dataset.char).join("");
            modalEx.classList.add("hidden");

            const res = await apiExchange(discard);
            if (res.status === "ok") {
                addHistoryRow("Player", `(Exchanged ${discard})`, "-", 0, gameState.player_score);
                const fresh = await apiGetState();
                renderState(fresh);
                setTimeout(triggerAiTurn, 500);
            }
        });
    }

    // Dictionary Lookup (guarded)
    const btnDict = document.getElementById("btn-dict-check");
    if (btnDict) {
        btnDict.addEventListener("click", async () => {
            const input = document.getElementById("dict-input");
            const val = input ? input.value.trim().toUpperCase() : "";
            if (!val) return;

            const res = await apiCheckWord(val);
            const card = document.getElementById("dict-result");
            if (card) {
                card.classList.remove("hidden", "valid", "invalid");
                if (res.valid) {
                    card.classList.add("valid");
                    card.textContent = `\u2713 "${val}" IS in the official tournament lexicon (SOWPODS)!`;
                } else {
                    card.classList.add("invalid");
                    card.textContent = `\u2717 "${val}" is NOT a valid Scrabble word.`;
                }
            }
        });
    }

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(btn.dataset.tab);
            if (targetPane) targetPane.classList.add("active");
        });
    });
}

// App Boot
async function startApp() {
    try {
        initBoardGrid();
        setupEvents();
        updateOpponentHeader();
    } catch (err) {
        console.error("UI setup error:", err);
    }

    try {
        const state = await apiGetState();
        renderState(state);
    } catch (e) {
        console.error("Could not fetch initial engine state:", e);
        if (statusMsgEl) statusMsgEl.textContent = "Connecting to native C11 Tessera engine...";
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startApp);
} else {
    startApp();
}
