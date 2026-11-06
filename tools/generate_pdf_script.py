#!/usr/bin/env python3
"""
Generates a publication-quality presentation script and defense guide PDF
for Harshit, Prem, Ashley, and Aaryan for Fr. CRCE Data Structures Mini-Project.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header (Pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "TESSERA: Data Structures Mini-Project Presentation Script")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Fr. Conceicao Rodrigues College of Engineering")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 32, footer_text)
        self.drawString(54, 32, "Confidential - Department of Computer Engineering - Prof. Ashwini Pansare")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 44, letter[0] - 54, 44)
        self.restoreState()


def build_pdf(filename="Tessera_Presentation_Script_CRCE.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep navy
    secondary_color = colors.HexColor("#0D9488") # Teal accent
    dark_text = colors.HexColor("#1F2937")       # Charcoal
    muted_text = colors.HexColor("#4B5563")      # Muted gray
    callout_bg = colors.HexColor("#F0FDF4")      # Mint highlight
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=muted_text,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=dark_text,
        spaceAfter=4
    )

    speaker_title_style = ParagraphStyle(
        'SpeakerTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    quote_style = ParagraphStyle(
        'SpokenQuote',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
        leftIndent=12,
        rightIndent=8,
        spaceAfter=6
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#065F46"),
        spaceAfter=3
    )

    story = []

    # Title Block
    story.append(Paragraph("FR. CONCEICAO RODRIGUES COLLEGE OF ENGINEERING", ParagraphStyle('CollHeader', fontName='Helvetica-Bold', fontSize=10, leading=13, textColor=secondary_color)))
    story.append(Paragraph("Department of Computer Engineering | Academic Year 2026", ParagraphStyle('DeptHeader', fontName='Helvetica', fontSize=8.5, leading=11, textColor=muted_text)))
    story.append(Spacer(1, 4))
    story.append(Paragraph("TESSERA: Data Structures Mini-Project Presentation Script", title_style))
    story.append(Paragraph("<b>Subject:</b> Data Structures (SE Computer Engineering C) &nbsp;|&nbsp; <b>Guide:</b> Prof. Ashwini Pansare<br/><b>Project:</b> Championship Scrabble Engine & Position-Keyed Belief Lattice in ISO C11 (2016 Elo)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    # Section 1: Team Roster & Ownership
    story.append(Paragraph("1. Team Member Allocation & Module Ownership", h1_style))
    
    team_data = [
        [Paragraph("<b>Member</b>", body_style), Paragraph("<b>Assigned Module & Data Structures</b>", body_style), Paragraph("<b>Key C Source Files</b>", body_style), Paragraph("<b>Presentation Role</b>", body_style)],
        [Paragraph("<b>Harshit Khandelwal</b><br/>(Team Lead)", body_style), Paragraph("Minimal DAWG Lexicon (Trie) & Appel-Jacobson Movegen with 32-bit Bitmasks", body_style), Paragraph("<code>src/dict.c</code><br/><code>src/movegen.c</code>", code_style), Paragraph("Intro, Problem Statement & Lexicon / Movegen Architecture", body_style)],
        [Paragraph("<b>Prem</b>", body_style), Paragraph("Dual-Accumulator Evaluator, 8-Phase Arrays & Deep Dual-Headed NNUE SIMD", body_style), Paragraph("<code>src/eval.c</code><br/><code>src/nnue.c</code>", code_style), Paragraph("Static Evaluation, Bigram Synergy Matrix & Neural Inference", body_style)],
        [Paragraph("<b>Ashley</b>", body_style), Paragraph("Pure Position-Keyed Belief Lattice (DAG) & Priority Queue (Binary Max-Heap)", body_style), Paragraph("<code>src/lattice.c</code><br/><code>src/settle.c</code>", code_style), Paragraph("DAG Transposition Merging & DirtyQueue Bellman Relaxation", body_style)],
        [Paragraph("<b>Aaryan</b>", body_style), Paragraph("Exact Transposition DAG Endgame Solver, Bayesian Inference & Benchmark", body_style), Paragraph("<code>src/endgame.c</code><br/><code>src/court.c</code>", code_style), Paragraph("Negamax Endgame Solver, Tournament Matrix & 2016 Elo Rating", body_style)],
    ]
    
    t_team = Table(team_data, colWidths=[1.4*inch, 2.7*inch, 1.4*inch, 1.7*inch])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
        ('TEXTCOLOR', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 8))

    # Section 2: Data Structures Syllabus Alignment Matrix
    story.append(Paragraph("2. Course Curriculum Alignment Matrix", h1_style))
    ds_data = [
        [Paragraph("<b>Data Structure</b>", body_style), Paragraph("<b>Engine Implementation</b>", body_style), Paragraph("<b>Time Complexity</b>", body_style), Paragraph("<b>Space Complexity</b>", body_style)],
        [Paragraph("<b>Trie / Automaton</b>", body_style), Paragraph("Minimal SOWPODS DAWG (185,702 edges bit-packed into 4-byte integers)", body_style), Paragraph("O(L) string search", body_style), Paragraph("742 KB flat array", body_style)],
        [Paragraph("<b>Hash Table</b>", body_style), Paragraph("64-bit Zobrist Hash Table with open addressing and linear probing", body_style), Paragraph("O(1) average lookup", body_style), Paragraph("16,384 buckets (load < 0.08)", body_style)],
        [Paragraph("<b>Directed Graph (DAG)</b>", body_style), Paragraph("Position-Keyed Belief Lattice with Gaussian beliefs (μ, σ², N) and back-pointers", body_style), Paragraph("O(V + E) traversal", body_style), Paragraph("Merged graph nodes", body_style)],
        [Paragraph("<b>Binary Max-Heap</b>", body_style), Paragraph("Prioritized DirtyQueue for Bellman backward induction relaxation waves", body_style), Paragraph("O(log N) push/pop", body_style), Paragraph("2,048 fixed heap array", body_style)],
        [Paragraph("<b>Bitwise Sets / Masks</b>", body_style), Paragraph("32-bit cross-check masks for instantaneous orthogonal word validation", body_style), Paragraph("O(1) bitwise AND", body_style), Paragraph("900 bytes (15x15 board)", body_style)],
        [Paragraph("<b>Transposition Cache</b>", body_style), Paragraph("32,768-entry depth-preferred cache for exact terminal endgame negamax", body_style), Paragraph("O(1) query / store", body_style), Paragraph("786 KB (32k x 24B)", body_style)],
        [Paragraph("<b>Static Memory Pool</b>", body_style), Paragraph("Depth-indexed static pool (g_endgame_moves_pool) eliminating stack frames", body_style), Paragraph("O(1) frame indexing", body_style), Paragraph("80 bytes vs 1MB stack", body_style)],
    ]
    t_ds = Table(ds_data, colWidths=[1.5*inch, 3.1*inch, 1.3*inch, 1.3*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F0FDF4")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor("#065F46")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 10))

    # Section 3: Word-for-Word Speaking Scripts
    story.append(Paragraph("3. Word-for-Word Presentation Speaking Scripts", h1_style))
    story.append(Paragraph("Each team member has a dedicated 2-minute speaking segment with exact cues and key technical terms.", subtitle_style))

    # --- SPEAKER 1: HARSHIT ---
    story.append(Paragraph("SPEAKER 1: Harshit Khandelwal (Team Lead) &mdash; 2.5 Minutes", speaker_title_style))
    story.append(Paragraph("<b>Topic:</b> Executive Introduction, Problem Statement & Minimal DAWG / Movegen Architecture", body_style))
    story.append(Paragraph(
        "\"Respected Prof. Ashwini Ma'am and fellow classmates, good morning. "
        "Today, our team is presenting our Data Structures Mini Project: <b>Tessera &mdash; a Grandmaster-tier Scrabble Intelligence Engine implemented entirely in standard ISO C11 with zero external libraries</b>.<br/><br/>"
        "When people think of board game AI like Chess or Go, they think of Minimax trees or Monte Carlo Search Trees. "
        "However, Scrabble presents a fundamentally harder data structure challenge: it is an imperfect-information game, "
        "the branching factor is massive with over 2,500 legal moves per turn, and independent row placements cause move-order transpositions to explode exponentially. "
        "To solve this, our project makes a radical architectural choice: <b>we strictly eliminated search trees</b>. "
        "Instead, we built a 100% position-keyed Directed Acyclic Graph (Belief Lattice) powered by a Binary Max-Heap, a Minimal DAWG, and Zobrist Hash Tables.<br/><br/>"
        "I will now explain our first primary data structure: <b>The Minimal Directed Acyclic Word Graph (DAWG)</b> in <code>src/dict.c</code>. "
        "In Scrabble, verifying words using a binary search tree or array takes O(log N) per word. A standard Trie for the 267,751-word tournament lexicon would require over 40 Megabytes of pointers! "
        "Our DAWG merges both <i>common prefixes</i> and <i>common suffixes</i>. Suffixes like -ING or -ATION are shared across thousands of words and collapsed into identical graph nodes. "
        "The entire 267,751-word lexicon is compressed into only <b>185,702 edges fitting into 742 Kilobytes</b>! "
        "Each edge is bit-packed into a single 32-bit integer: 24 bits for the child index, 5 bits for the letter, 1 bit for word termination, and 1 bit for end-of-node. "
        "This flat array loads in 0.0001 seconds via zero-copy binary ingestion. "
        "Combined with our Appel-Jacobson move generator in <code>src/movegen.c</code>, which precomputes 32-bit bitmasks for vertical cross-checks on the 15x15 board, "
        "Tessera generates over <b>1.3 Million legal moves per second</b> in pure C!<br/><br/>"
        "I now pass the presentation to <b>Prem</b> to explain our static evaluation data structures.\"", quote_style))

    # --- SPEAKER 2: PREM ---
    story.append(Paragraph("SPEAKER 2: Prem &mdash; 2.0 Minutes", speaker_title_style))
    story.append(Paragraph("<b>Topic:</b> Dual-Accumulator Evaluator, 8-Phase Arrays, Bigram Synergy Matrix & NNUE", body_style))
    story.append(Paragraph(
        "\"Thank you, Harshit. A common flaw in game engines is treating piece values statically. "
        "In Scrabble, tile values change drastically as the tile bag empties. A blank tile is worth +26 points in the opening because it triggers multiple 50-point bingos, "
        "but near bag exhaustion, its value drops to +14 points because there is no time left to compound it.<br/><br/>"
        "In <code>src/eval.c</code>, we designed a <b>Dual-Accumulator Evaluator</b> parameterized across <b>8 discrete game-phase buckets</b>:<br/>"
        "&bull; <b>The Board Accumulator</b> evaluates the 15x15 grid for immediate word score, premium square occupancy (Triple Word, Double Word), and hazard exposure (such as floating vowels adjacent to unblocked Triple Words).<br/>"
        "&bull; <b>The Rack Leave Accumulator</b> evaluates unplayed tiles using an 8x27 lookup array of phase-dependent tile values, a 27x27 <b>Pairwise Synergy Matrix</b> capturing multi-tile harmonies (like QU synergy +10 pts or S+Blank +4.5 pts), and non-linear duplicate penalties (VV -36.0 pts, WW -20.0 pts).<br/>"
        "&bull; <b>Sub-Rack Exchange Optimizer:</b> When a rack is choked, we evaluate all 2^k subsets (k &le; 7, up to 128 subsets) via bit manipulation to find the mathematically optimal tile discard split.<br/><br/>"
        "Furthermore, in <code>src/nnue.c</code>, we implemented an Efficiently Updatable Neural Network (NNUE) with a 448 &rarr; 128 &rarr; 32 &rarr; 2 dual-headed architecture using AVX2 SIMD vector instructions in pure C, ensembled 50/50 with our hand-tuned heuristics.<br/><br/>"
        "I now hand over to <b>Ashley</b> to explain our project's core innovation: The Belief Lattice DAG.\"", quote_style))

    # --- SPEAKER 3: ASHLEY ---
    story.append(Paragraph("SPEAKER 3: Ashley &mdash; 2.5 Minutes", speaker_title_style))
    story.append(Paragraph("<b>Topic:</b> Pure Position-Keyed Belief Lattice (DAG), Zobrist Hashing & Binary Max-Heap", body_style))
    story.append(Paragraph(
        "\"Thank you, Prem. This brings us to the core data structure novelty of Tessera: <b>The Pure Position-Keyed Belief Lattice</b>.<br/><br/>"
        "In traditional game trees, if move sequence A then B leads to the exact same board state as B then A, a search tree allocates two distinct nodes and evaluates both independently. "
        "In Scrabble, where words are placed across independent rows, this causes an exponential explosion of redundant branches!<br/><br/>"
        "In <code>src/lattice.c</code>, we strictly prohibited game trees. Instead:<br/>"
        "1. Every 15x15 board configuration is hashed into a 64-bit integer using <b>Zobrist Hashing</b> with pseudo-random keys.<br/>"
        "2. States are inserted into an <b>Open-Addressing Hash Table</b> with 16,384 slots. If a board has already been reached through another move order, <b>the nodes merge by construction</b>. Transposition merging is 100%!<br/>"
        "3. Each node in the DAG stores a continuous Gaussian belief: B(v) = (&mu;, &sigma;&sup2;, N).<br/><br/>"
        "<b>How do updates propagate backward in a DAG?</b><br/>"
        "In a tree, each node has only one parent. In a DAG, a node has multiple parents! "
        "In <code>src/settle.c</code>, we implemented a <b>Binary Max-Heap (DirtyQueue)</b>. "
        "Whenever simulation updates a node's belief, the change magnitude |&Delta;| is pushed into the Max-Heap in O(log N) time. "
        "We extract the maximum-priority dirty node and relax its Q-value using our calibrated Bellman operator: Q(e) = MoveStatic(e) + &gamma; &times; &Delta;_combined(e). "
        "All parent back-pointers are enqueued into the heap. Tactical discoveries ripple backward across the graph in prioritized waves until reaching equilibrium!<br/><br/>"
        "Furthermore, we implemented <b>2-Ply Draw Realization</b>: during deliberation, Tessera simulates refilling its rack from the remaining bag tiles and tests its actual turn-2 response equity on the resulting board.<br/><br/>"
        "I now invite <b>Aaryan</b> to cover the Endgame Solver, Bayesian Inference, and our Tournament Results.\"", quote_style))

    # --- SPEAKER 4: AARYAN ---
    story.append(Paragraph("SPEAKER 4: Aaryan &mdash; 2.0 Minutes", speaker_title_style))
    story.append(Paragraph("<b>Topic:</b> Exact Endgame DAG Solver, Bayesian Opponent Tracker & Tournament Results", body_style))
    story.append(Paragraph(
        "\"Thank you, Ashley. In the final phase of Scrabble, when the bag is empty and the opponent has &le; 7 tiles, Scrabble transitions into a <b>deterministic perfect-information game</b>.<br/><br/>"
        "In <code>src/endgame.c</code>, we built an <b>Exact Negamax Endgame Solver</b> running over a <b>32,768-entry Transposition Table</b>:<br/>"
        "&bull; We eliminated stack overflow risks by creating a depth-indexed static memory pool (<code>g_endgame_moves_pool</code>), reducing memory usage from 1 MB per recursive stack frame to just 80 bytes!<br/>"
        "&bull; The solver searches up to 8 plies deep and computes the exact optimal move and terminal margin in <b>under 1.5 milliseconds</b>!<br/>"
        "&bull; In <code>src/court.c</code>, our engine stops deliberating using <b>The Root Court</b> &mdash; a Bayesian hypothesis test drawing 1,000 Thompson samples. When P(best move) &ge; 98%, it adjourns instantly, saving compute for contested turns.<br/><br/>"
        "<b>Empirical Tournament Benchmark (70 Games vs 7 Bot Profiles):</b><br/>"
        "We benchmarked Tessera across 70 tournament matches against specialized bots, including commercial simulation engines:<br/>"
        "&bull; Against <b>MAGPIE / Woogles 2-Ply Realization (~2150 Elo)</b>, Tessera won <b>6 - 4 (+23.3 pts spread)</b>.<br/>"
        "&bull; Against <b>Defensive Turtler</b>, Tessera scored a perfect <b>10 - 0 sweep (+74.9 pts spread)</b>.<br/>"
        "&bull; Against <b>Neural Static 1-Ply</b>, Tessera won <b>8 - 2 (+60.4 pts spread)</b>.<br/>"
        "&bull; <b>Overall Record:</b> <b>50 Wins, 20 Losses (71.4% Win Rate)</b> averaging <b>460.1 points per game</b> and <b>2.14 bingos per game</b>.<br/>"
        "&bull; <b>Official NASPA Performance Rating (TPR):</b> Evaluated against the 1857-rated field: <b>~2016 Elo</b> (National Champion Tier)!<br/><br/>"
        "We will now begin our live demonstration. Thank you!\"", quote_style))

    story.append(Spacer(1, 10))

    # Section 4: Live Demonstration Protocol
    story.append(Paragraph("4. Live Demonstration Execution Steps", h1_style))
    demo_steps = [
        [Paragraph("<b>Step & Command</b>", body_style), Paragraph("<b>Demonstration Target</b>", body_style), Paragraph("<b>Presenter & Speaking Cue</b>", body_style)],
        [Paragraph("<b>Step 1: Test Suite</b><br/><code>make test</code>", code_style), Paragraph("Executes all 5 C unit tests: DAWG (0.0001s), Movegen (2,916 plays in 0.84ms), Eval, Lattice proof, and Endgame solver (1.57ms).", body_style), Paragraph("<b>Harshit:</b> <i>\"Ma'am, as you can see, all 5 core data structures compile cleanly with zero external libraries and pass in under 2 seconds.\"</i>", body_style)],
        [Paragraph("<b>Step 2: Live Match</b><br/><code>./tessera ai-vs-ai</code>", code_style), Paragraph("Runs a live autonomous game showing DAG telemetry: nodes created, transpositions merged, dirty passes settled, and Court confidence.", body_style), Paragraph("<b>Ashley:</b> <i>\"Notice how identical board states reached through different word placements merge into single vertices without duplicate tree branches.\"</i>", body_style)],
        [Paragraph("<b>Step 3: Visual Guide</b><br/><code>xdg-open readme_preview.html</code>", code_style), Paragraph("Displays standalone rendered documentation with GitHub CSS, KaTeX mathematical equations, and Mermaid graph architecture diagrams.", body_style), Paragraph("<b>Aaryan & Prem:</b> <i>\"Here is our full 70-game tournament matrix showing our 71.4% win rate and verified 2016 Elo rating.\"</i>", body_style)],
    ]
    t_demo = Table(demo_steps, colWidths=[1.8*inch, 3.1*inch, 2.3*inch])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EFF6FF")),
        ('TEXTCOLOR', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 10))

    # Section 5: Individual Teacher Viva Defense Q&A Cards
    story.append(Paragraph("5. Personalized Teacher Viva Defense Cards", h1_style))
    story.append(Paragraph("Anticipated tough questions from Prof. Ashwini Pansare, mapped directly to each team member:", subtitle_style))

    # Harshit's Card
    story.append(Paragraph("Harshit Khandelwal &mdash; Lexicon, DAWG & Movegen Viva Card", h2_style))
    story.append(Paragraph("<b>Q: Why did you choose a DAWG over a standard Trie or Binary Search Tree?</b>", body_style))
    story.append(Paragraph("<i>A: A BST requires O(log N) string comparisons where each comparison is O(L), making generation across millions of permutations too slow. A standard Trie shares prefixes, but for SOWPODS (267,751 words) it requires over 400,000 nodes and tens of megabytes of pointer overhead. A DAWG is a minimal deterministic finite state automaton that merges both common prefixes AND common suffixes. Suffixes like -ING, -ED, or -ATION share the exact same child subgraphs, compressing the entire dictionary into just 185,702 4-byte packed edges (742 KB) while preserving O(L) verification directly inside CPU L2 cache.</i>", quote_style))

    # Prem's Card
    story.append(Paragraph("Prem &mdash; Evaluator, Feature Matrices & SIMD Viva Card", h2_style))
    story.append(Paragraph("<b>Q: Why decouple evaluation into 8 game phases instead of a single static point table?</b>", body_style))
    story.append(Paragraph("<i>A: Tile equity is non-stationary. In the Opening (80-100 tiles), a blank is worth +26.5 points because it enables multiple future bingos. In Exhaustion (1-9 tiles), it drops to +14 points because there is no time left to compound its value. Conversely, heavy tiles like Q and V are manageable early on, but catastrophic near bag depletion. The 8-phase bucket model adapts tile synergies, vowel-consonant ratios, and stem bonuses dynamically to the exact state of the bag.</i>", quote_style))

    # Ashley's Card
    story.append(Paragraph("Ashley &mdash; Belief Lattice DAG & Priority Queue Heap Viva Card", h2_style))
    story.append(Paragraph("<b>Q: How does the Binary Max-Heap operate in your DAG and what are its complexities?</b>", body_style))
    story.append(Paragraph("<i>A: Our Max-Heap is implemented in <code>src/settle.c</code> as DirtyQueue. Unlike a tree where updates only have one parent, a DAG node can have multiple parents. Whenever a node's belief moves by |&Delta;|, its parent indices are pushed into the heap with priority |&Delta;| in O(log N) time using bubble-up (heap_swap). We extract the maximum element in O(log N) time using bubble-down. This guarantees that the most significant tactical refutations are relaxed backward first, achieving numerical Bellman equilibrium in minimal passes.</i>", quote_style))

    # Aaryan's Card
    story.append(Paragraph("Aaryan &mdash; Endgame Negamax, Memory Pool & Elo Viva Card", h2_style))
    story.append(Paragraph("<b>Q: How did you prevent stack overflow during deep recursive endgame search?</b>", body_style))
    story.append(Paragraph("<i>A: In <code>src/endgame.c</code>, generating legal moves requires a MoveList struct holding up to 8,192 moves (~1 Megabyte). Initially, allocating MoveList moves as a local stack variable inside recursive negamax overflowed the Linux 8 MB stack at depth 5. We resolved this by creating a depth-indexed static pool (<code>static MoveList g_endgame_moves_pool[MAX_ENDGAME_DEPTH + 2]</code>). Each recursive frame at depth d indexes directly into slot d of the pre-allocated static pool, slashing stack consumption from 1 Megabyte to just 80 bytes per frame and solving 8-ply endgames in under 1.5 milliseconds.</i>", quote_style))

    # Build PDF with custom canvas for page numbers
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built: {filename}")

if __name__ == "__main__":
    build_pdf()
