/**
 * TESSERA — Scrabble Engine Web UI Controller
 * Communicates with the native C engine via JSON REST API.
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

// State
let gameState = null;
let pendingPlaced = []; // Array of {row, col, letter, is_blank, original_rack_index}
let selectedRackIndex = -1;
let moveHistory = [];
let isAiVsAiRunning = false;
let aiVsAiTimer = null;

// DOM Elements
const boardEl = document.getElementById("scrabble-board");
const colLabelsEl = document.getElementById("col-labels");
const rowLabelsEl = document.getElementById("row-labels");
const rackEl = document.getElementById("player-rack");
const statusMsgEl = document.getElementById("status-message");
const scorePlayerEl = document.getElementById("score-player");
const scoreAiEl = document.getElementById("score-ai");
const bagCountEl = document.getElementById("bag-count");
const hintsTbodyEl = document.getElementById("hints-tbody");
const historyTbodyEl = document.getElementById("history-tbody");
const tileTrackerGridEl = document.getElementById("tile-tracker-grid");

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

            // Determine multiplier styling
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

// Render Board & Racks from State
function renderState(state) {
    gameState = state;

    scorePlayerEl.textContent = state.player_score;
    scoreAiEl.textContent = state.ai_score;
    bagCountEl.textContent = state.bag_count;

    // Render cells
    const cells = boardEl.querySelectorAll(".cell");
    cells.forEach(cell => {
        const r = parseInt(cell.dataset.row);
        const c = parseInt(cell.dataset.col);

        // Clear existing tiles
        const oldTile = cell.querySelector(".tile");
        if (oldTile) oldTile.remove();

        // Check if committed in board state
        const boardCell = state.board[r][c];
        if (boardCell.tile) {
            const tileEl = createTileElement(boardCell.tile, boardCell.is_blank);
            cell.appendChild(tileEl);
        } else {
            // Check if pending user placement
            const pending = pendingPlaced.find(p => p.row === r && p.col === c);
            if (pending) {
                const tileEl = createTileElement(pending.letter, pending.is_blank, true);
                cell.appendChild(tileEl);
            }
        }
    });

    renderRack();
    renderTileTracker();

    // Update Phase Bucket & Openness Telemetry
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

function renderRack() {
    rackEl.innerHTML = "";
    if (!gameState || !gameState.player_rack) return;

    const availableTiles = [];
    const usedIndices = new Set(pendingPlaced.map(p => p.original_rack_index));

    for (let i = 0; i < gameState.player_rack.length; i++) {
        if (!usedIndices.has(i)) {
            availableTiles.push({ letter: gameState.player_rack[i], index: i });
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
            renderRack();
        });

        rackEl.appendChild(tileEl);
    });
}

function handleCellClick(r, c) {
    if (isAiVsAiRunning) return;

    // 1. If square already has a pending placed tile, return it to rack!
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

    // 3. If a rack tile is currently selected, place it here!
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

// Tile Tracker (Tab 3)
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

    // Subtract player rack
    for (const ch of gameState.player_rack) {
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

// Play Move Submission
async function submitPlayMove() {
    if (pendingPlaced.length === 0) {
        alert("Please place at least one tile on the board.");
        return;
    }

    // Determine orientation: all placed tiles must be in same row or col
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
        // If 1 tile placed, check adjacent board tiles to determine direction
        const r = pendingPlaced[0].row;
        const c = pendingPlaced[0].col;
        const hasH = (c > 0 && gameState.board[r][c - 1].tile) || (c < 14 && gameState.board[r][c + 1].tile);
        const hasV = (r > 0 && gameState.board[r - 1][c].tile) || (r < 14 && gameState.board[r + 1][c].tile);
        if (hasV && !hasH) dir = "V";
        else dir = "H";
    }

    // Find bounding word
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
        addHistoryRow("Player", fullWord, `${String.fromCharCode(65 + startC)}${startR + 1}`, res.score, res.player_score);
        statusMsgEl.textContent = `Player scored ${res.score} points with ${fullWord}! AI is thinking...`;

        if (res.metrics) {
            const elNodes = document.getElementById("telemetry-nodes");
            const elTrans = document.getElementById("telemetry-transpositions");
            const elPasses = document.getElementById("telemetry-dirty-passes");
            const elConf = document.getElementById("telemetry-court-conf");
            const elPalimp = document.getElementById("telemetry-palimpsest");
            if (elNodes) elNodes.textContent = res.metrics.nodes || 0;
            if (elTrans) elTrans.textContent = res.metrics.transpositions || 0;
            if (elPasses) elPasses.textContent = res.metrics.settle_passes || 0;
            if (elConf) elConf.textContent = ((res.metrics.confidence || 0) * 100).toFixed(1) + "%";
            if (elPalimp) elPalimp.textContent = (res.metrics.palimpsest_bias || 0).toFixed(2) + " pts";
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
            statusMsgEl.innerHTML = `<span style="color:#2ecc71;font-weight:bold;">👑 EXACT ENDGAME MINIMAX ACTIVE</span>: AI played ${res.move.word} (Proven Spread: ${res.move.equity >= 0 ? '+' : ''}${res.move.equity.toFixed(0)} pts). Your turn!`;
        }

        const freshState = await apiGetState();
        renderState(freshState);

        // Trigger AI reply
        setTimeout(triggerAiTurn, 600);
    } else {
        alert("Move rejected: " + res.message);
        statusMsgEl.textContent = "Move rejected: " + res.message;
    }
}

async function triggerAiTurn() {
    statusMsgEl.innerHTML = `<span class="spinner"></span> Tessera AI is calculating Monte Carlo simulations...`;
    const res = await apiAiMove();

    if (res.status === "ok") {
        if (res.move.type === 0) { // MOVE_PLAY
            const pos = `${String.fromCharCode(65 + res.move.col)}${res.move.row + 1}`;
            addHistoryRow("Tessera AI", res.move.word, pos, res.move.score, res.ai_score);
            statusMsgEl.textContent = `Tessera AI played ${res.move.word} for ${res.move.score} points (${res.time_ms.toFixed(1)} ms). Your turn!`;
        } else if (res.move.type === 1) {
            addHistoryRow("Tessera AI", "(Exchanged Tiles)", "-", 0, res.ai_score);
            statusMsgEl.textContent = `Tessera AI exchanged tiles. Your turn!`;
        } else {
            addHistoryRow("Tessera AI", "(Pass)", "-", 0, res.ai_score);
            statusMsgEl.textContent = `Tessera AI passed. Your turn!`;
        }

        if (res.metrics) {
            const elNodes = document.getElementById("telemetry-nodes");
            const elTrans = document.getElementById("telemetry-transpositions");
            const elPasses = document.getElementById("telemetry-dirty-passes");
            const elConf = document.getElementById("telemetry-court-conf");
            const elPalimp = document.getElementById("telemetry-palimpsest");
            if (elNodes) elNodes.textContent = res.metrics.nodes || 0;
            if (elTrans) elTrans.textContent = res.metrics.transpositions || 0;
            if (elPasses) elPasses.textContent = res.metrics.settle_passes || 0;
            if (elConf) elConf.textContent = ((res.metrics.confidence || 0) * 100).toFixed(1) + "%";
            if (elPalimp) elPalimp.textContent = (res.metrics.palimpsest_bias || 0).toFixed(2) + " pts";
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
            statusMsgEl.innerHTML = `<span style="color:#2ecc71;font-weight:bold;">👑 EXACT ENDGAME MINIMAX ACTIVE</span>: AI played ${res.move.word} (Proven Spread: ${res.move.equity >= 0 ? '+' : ''}${res.move.equity.toFixed(0)} pts). Your turn!`;
        }

        const freshState = await apiGetState();
        renderState(freshState);

        if (freshState.game_over) {
            statusMsgEl.textContent = `Game Over! Final: Player ${freshState.player_score} - AI ${freshState.ai_score}`;
        }
    } else {
        statusMsgEl.textContent = res.message || "AI encountered an error.";
    }
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

// AI vs AI Demonstration Mode
async function toggleAiVsAi() {
    const btn = document.getElementById("btn-ai-vs-ai");
    if (isAiVsAiRunning) {
        isAiVsAiRunning = false;
        clearInterval(aiVsAiTimer);
        btn.textContent = "⚡ AI vs AI Demo";
        btn.classList.remove("btn-accent");
        btn.classList.add("btn-primary");
        statusMsgEl.textContent = "AI vs AI match paused.";
    } else {
        isAiVsAiRunning = true;
        btn.textContent = "⏸ Pause AI Match";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-accent");

        runAiVsAiStep();
    }
}

async function runAiVsAiStep() {
    if (!isAiVsAiRunning) return;

    statusMsgEl.innerHTML = `<span class="spinner"></span> AI self-play thinking...`;
    const res = await apiAiMove();

    if (res.status === "ok") {
        if (res.move.type === 0) {
            const pos = `${String.fromCharCode(65 + res.move.col)}${res.move.row + 1}`;
            addHistoryRow("Tessera AI", res.move.word, pos, res.move.score, res.ai_score);
            statusMsgEl.textContent = `AI played ${res.move.word} for ${res.move.score} pts (${res.time_ms.toFixed(1)} ms).`;
        }
        const fresh = await apiGetState();
        renderState(fresh);

        if (fresh.game_over) {
            isAiVsAiRunning = false;
            document.getElementById("btn-ai-vs-ai").textContent = "⚡ AI vs AI Demo";
            statusMsgEl.textContent = `Match Finished! Final Score: Player ${fresh.player_score} - AI ${fresh.ai_score}`;
            return;
        }

        aiVsAiTimer = setTimeout(runAiVsAiStep, 1000);
    }
}

// Event Listeners
function setupEvents() {
    document.getElementById("btn-new-game").addEventListener("click", async () => {
        pendingPlaced = [];
        moveHistory = [];
        historyTbodyEl.innerHTML = `<tr><td colspan="6" class="placeholder-text">No moves played yet.</td></tr>`;
        const state = await apiNewGame();
        renderState(state);
        statusMsgEl.textContent = "New match started! Your turn.";
    });

    document.getElementById("btn-ai-vs-ai").addEventListener("click", toggleAiVsAi);

    document.getElementById("btn-recall-tiles").addEventListener("click", () => {
        pendingPlaced = [];
        selectedRackIndex = -1;
        renderState(gameState);
    });

    document.getElementById("btn-shuffle-rack").addEventListener("click", () => {
        if (!gameState || !gameState.player_rack) return;
        const arr = gameState.player_rack.split("");
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        gameState.player_rack = arr.join("");
        renderRack();
    });

    document.getElementById("btn-play-move").addEventListener("click", submitPlayMove);
    document.getElementById("btn-ai-play").addEventListener("click", triggerAiTurn);

    document.getElementById("btn-pass").addEventListener("click", async () => {
        if (confirm("Are you sure you want to pass your turn?")) {
            await apiPass();
            addHistoryRow("Player", "(Pass)", "-", 0, gameState.player_score);
            statusMsgEl.textContent = "Player passed. AI is thinking...";
            setTimeout(triggerAiTurn, 500);
        }
    });

    document.getElementById("btn-get-hint").addEventListener("click", async () => {
        const loading = document.getElementById("engine-hint-loading");
        loading.classList.remove("hidden");
        const res = await apiGetHint();
        loading.classList.add("hidden");

        if (res.status === "ok" && res.hints) {
            if (res.phase) {
                const elPhase = document.getElementById("telemetry-phase-bucket");
                if (elPhase) elPhase.textContent = res.phase;
            }
            if (res.openness) {
                const elOpen = document.getElementById("telemetry-openness");
                if (elOpen) elOpen.textContent = res.openness;
            }
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
                    <td>${h.leave || "—"}</td>
                    <td>${h.equity.toFixed(1)}</td>
                `;
                hintsTbodyEl.appendChild(tr);
            });
            // Switch to analysis tab
            document.querySelector('[data-tab="tab-engine"]').click();
        }
    });

    // Exchange Modal
    const modalEx = document.getElementById("modal-exchange");
    const modalRackEl = document.getElementById("modal-exchange-rack");
    document.getElementById("btn-exchange").addEventListener("click", () => {
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

    document.getElementById("btn-exchange-cancel").addEventListener("click", () => {
        modalEx.classList.add("hidden");
    });

    document.getElementById("btn-exchange-confirm").addEventListener("click", async () => {
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

    // Dictionary Lookup
    document.getElementById("btn-dict-check").addEventListener("click", async () => {
        const input = document.getElementById("dict-input");
        const val = input.value.trim().toUpperCase();
        if (!val) return;

        const res = await apiCheckWord(val);
        const card = document.getElementById("dict-result");
        card.classList.remove("hidden", "valid", "invalid");
        if (res.valid) {
            card.classList.add("valid");
            card.textContent = `✓ "${val}" IS in the official SOWPODS tournament dictionary!`;
        } else {
            card.classList.add("invalid");
            card.textContent = `✗ "${val}" is NOT a valid Scrabble word.`;
        }
    });

    // Tab Navigation
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");
        });
    });
}

// App Boot
async function startApp() {
    initBoardGrid();
    setupEvents();
    try {
        const state = await apiGetState();
        renderState(state);
    } catch (e) {
        console.error("Could not fetch state:", e);
        statusMsgEl.textContent = "Connecting to native engine...";
    }
}

window.addEventListener("DOMContentLoaded", startApp);
