#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

#include "board.h"
#include "boardparameters.h"
#include "test/trademarkedboards.h"
#include "datamanager.h"
#include "lexiconparameters.h"
#include "strategyparameters.h"
#include "game.h"
#include "bogowinplayer.h"

#include "quackle_bridge.h"

static Quackle::Move bridge_to_quackle_move(const BridgeMove &bm, const Quackle::Board &qb) {
    if (bm.type == 2) {
        return Quackle::Move::createPassMove();
    }
    if (bm.type == 1) {
        Quackle::LetterString qtiles;
        for (int i = 0; i < bm.num_placed; i++) {
            if (bm.placed[i].is_blank) {
                qtiles += QUACKLE_BLANK_MARK;
            } else {
                Quackle::Letter qlet = QUACKLE_ALPHABET_PARAMETERS->encode(UVString(1, bm.placed[i].letter))[0];
                qtiles += qlet;
            }
        }
        return Quackle::Move::createExchangeMove(qtiles, false);
    }

    int r = bm.row;
    int c = bm.col;
    bool horizontal = bm.horizontal;
    int len = (int)strlen(bm.word);

    Quackle::LetterString qtiles;
    for (int k = 0; k < len; k++) {
        int cur_r = horizontal ? r : r + k;
        int cur_c = horizontal ? c + k : c;

        if (qb.letter(cur_r, cur_c) != QUACKLE_NULL_MARK) {
            qtiles += QUACKLE_PLAYED_THRU_MARK;
        } else {
            bool is_blank = false;
            for (int p = 0; p < bm.num_placed; p++) {
                if (bm.placed[p].row == cur_r && bm.placed[p].col == cur_c) {
                    is_blank = bm.placed[p].is_blank;
                    break;
                }
            }
            Quackle::Letter qlet = QUACKLE_ALPHABET_PARAMETERS->encode(UVString(1, (char)toupper(bm.word[k])))[0];
            if (is_blank) {
                qtiles += QUACKLE_ALPHABET_PARAMETERS->setBlankness(qlet);
            } else {
                qtiles += qlet;
            }
        }
    }

    Quackle::Move qm = Quackle::Move::createPlaceMove(r, c, horizontal, qtiles);
    qm.score = qb.score(qm, &qm.isBingo);
    return qm;
}

struct MatchResult {
    int tessera_score;
    int quackle_score;
    int tessera_bingos;
    int quackle_bingos;
    double game_seconds;
    int first_player; // 0: Tessera, 1: Quackle
};

static MatchResult play_game(void *engine, const std::string &opponent_type,
                             int game_idx, int seconds_per_turn, int tessera_delib, bool verbose) {
    MatchResult res = {0, 0, 0, 0, 0.0, game_idx % 2};
    auto start_time = std::chrono::high_resolution_clock::now();

    Quackle::Game game;
    Quackle::PlayerList players;

    int tess_id = (game_idx % 2 == 0) ? 0 : 1;
    int quack_id = (game_idx % 2 == 0) ? 1 : 0;

    int comp_player_id = (opponent_type == "speedy") ? 1 : 110;
    Quackle::Player p_tess(MARK_UV("Tessera"), Quackle::Player::HumanPlayerType, 0);
    Quackle::Player p_quack(MARK_UV("Quackle"), Quackle::Player::ComputerPlayerType, comp_player_id);

    if (opponent_type == "speedy") {
        p_quack.setComputerPlayer(new Quackle::StaticPlayer());
    } else {
        Quackle::SmartBogowin *bogowin = new Quackle::SmartBogowin();
        Quackle::ComputerParameters params;
        params.secondsPerTurn = seconds_per_turn;
        bogowin->setParameters(params);
        bogowin->setThreadCount(1, Quackle::ThreadQoS::Balanced);
        p_quack.setComputerPlayer(bogowin);
    }

    if (tess_id == 0) {
        players.push_back(p_tess);
        players.push_back(p_quack);
    } else {
        players.push_back(p_quack);
        players.push_back(p_tess);
    }

    game.setPlayers(players);
    game.associateKnownComputerPlayers();
    game.setDefaultComputerPlayer(new Quackle::StaticPlayer());
    game.addPosition();

    void *tracker = tessera_create_tracker();
    int turn_count = 0;

    while (!game.currentPosition().gameOver()) {
        turn_count++;

        if (game.currentPosition().currentPlayer().type() == Quackle::Player::HumanPlayerType) {
            char board_tiles[15][15];
            bool board_is_blank[15][15];
            const Quackle::Board &qb = game.currentPosition().board();

            for (int r = 0; r < 15; r++) {
                for (int c = 0; c < 15; c++) {
                    Quackle::Letter qlet = qb.letter(r, c);
                    if (qlet != QUACKLE_NULL_MARK) {
                        UVString uvs = QUACKLE_ALPHABET_PARAMETERS->userVisible(qlet);
                        board_tiles[r][c] = (char)uvs[0];
                        board_is_blank[r][c] = qb.isBlank(r, c);
                    } else {
                        board_tiles[r][c] = 0;
                        board_is_blank[r][c] = false;
                    }
                }
            }

            std::string rack_str;
            const Quackle::LetterString &qtiles = game.currentPosition().currentPlayer().rack().tiles();
            for (size_t i = 0; i < qtiles.length(); i++) {
                Quackle::Letter c = qtiles[i];
                if (QUACKLE_ALPHABET_PARAMETERS->isBlankLetter(c) || c == QUACKLE_BLANK_MARK) {
                    rack_str.push_back('?');
                } else {
                    UVString uvs = QUACKLE_ALPHABET_PARAMETERS->userVisible(c);
                    rack_str.push_back((char)uvs[0]);
                }
            }

            int bag_count = (int)game.currentPosition().bag().size();
            int t_curr = game.currentPosition().players()[tess_id].score();
            int q_curr = game.currentPosition().players()[quack_id].score();
            int score_diff = t_curr - q_curr;

            BridgeMove bm = tessera_compute_move(engine, board_tiles, board_is_blank,
                                                rack_str.c_str(), bag_count, score_diff,
                                                tessera_delib, tracker);

            if (bm.type == 0 && bm.num_placed == 7) {
                res.tessera_bingos++;
            }

            Quackle::Move qm = bridge_to_quackle_move(bm, qb);
            if (verbose) {
                std::cout << "  Turn " << std::setw(2) << turn_count << " [Tessera]: " 
                          << qm.toString() << " (+" << qm.score << " pts) [Bag: " 
                          << bag_count << "]" << std::endl;
            }
            game.commitMove(qm);
        } else {
            Quackle::Move compMove = game.haveComputerPlay();
            if (compMove.isBingo) {
                res.quackle_bingos++;
            }
            if (verbose) {
                std::cout << "  Turn " << std::setw(2) << turn_count << " [Quackle]: " 
                          << compMove.toString() << " (+" << compMove.score << " pts) [Bag: " 
                          << game.currentPosition().bag().size() << "]" << std::endl;
            }
        }
    }

    tessera_destroy_tracker(tracker);

    Quackle::PlayerList final_players = game.currentPosition().endgameAdjustedScores();
    for (const auto &p : final_players) {
        if (p.id() == tess_id) res.tessera_score = p.score();
        if (p.id() == quack_id) res.quackle_score = p.score();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    res.game_seconds = std::chrono::duration<double>(end_time - start_time).count();
    return res;
}

int main(int argc, char **argv) {
    pthread_attr_t default_attr;
    pthread_attr_init(&default_attr);
    pthread_attr_setstacksize(&default_attr, 16 * 1024 * 1024);
    pthread_setattr_default_np(&default_attr);
    pthread_attr_destroy(&default_attr);

    int total_games = 20;
    std::string opponent = "smartbogowin";
    int seconds_per_turn = 1;
    int tessera_delib = 20;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--games" && i + 1 < argc) total_games = std::stoi(argv[++i]);
        else if (arg == "--opponent" && i + 1 < argc) opponent = argv[++i];
        else if (arg == "--seconds" && i + 1 < argc) seconds_per_turn = std::stoi(argv[++i]);
        else if (arg == "--delib" && i + 1 < argc) tessera_delib = std::stoi(argv[++i]);
        else if (arg == "--verbose") verbose = true;
        else if (arg == "-v") verbose = true;
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "     TESSERA vs. QUACKLE ENGINE OFFICIAL HEAD-TO-HEAD TOURNAMENT BENCHMARK     " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Lexicon:      SOWPODS (Collins Scrabble Tournament Lexicon, 267,751 words)" << std::endl;
    std::cout << "  Tessera:      Distributional Belief-Lattice Engine (Deliberation = " << tessera_delib << " steps)" << std::endl;
    std::cout << "  Opponent:     Quackle " << (opponent == "speedy" ? "SpeedyPlayer (Heuristic Baseline)" : "SmartBogowin (Monte Carlo Sim Engine)") << std::endl;
    std::cout << "  Quackle Time: " << seconds_per_turn << " sec/turn" << std::endl;
    std::cout << "  Total Games:  " << total_games << " matched games (alternating opening moves)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    Quackle::DataManager dataManager;
    dataManager.setAppDataDirectory("/tmp/quackle/data");
    dataManager.setBoardParameters(new ScrabbleBoard());
    dataManager.lexiconParameters()->loadDawg("/tmp/quackle/data/lexica/sowpods.dawg");
    dataManager.strategyParameters()->initialize("sowpods");

    void *engine = tessera_create_engine("data/dawg.bin");
    if (!engine) {
        std::cerr << "Error: Failed to load Tessera DAWG at data/dawg.bin" << std::endl;
        return 1;
    }

    int tess_wins = 0, quack_wins = 0, ties = 0;
    int total_tess_score = 0, total_quack_score = 0;
    int total_tess_bingos = 0, total_quack_bingos = 0;
    double total_match_time = 0;

    std::cout << std::left << std::setw(6) << "Game"
              << std::setw(12) << "First"
              << std::setw(10) << "Tessera"
              << std::setw(10) << "Quackle"
              << std::setw(10) << "Spread"
              << std::setw(10) << "T-Bingos"
              << std::setw(10) << "Q-Bingos"
              << std::setw(10) << "Outcome"
              << std::setw(10) << "Time" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    for (int g = 0; g < total_games; g++) {
        MatchResult res = play_game(engine, opponent, g, seconds_per_turn, tessera_delib, verbose);

        total_tess_score += res.tessera_score;
        total_quack_score += res.quackle_score;
        total_tess_bingos += res.tessera_bingos;
        total_quack_bingos += res.quackle_bingos;
        total_match_time += res.game_seconds;

        std::string outcome;
        int spread = res.tessera_score - res.quackle_score;
        if (spread > 0) {
            tess_wins++;
            outcome = "WIN";
        } else if (spread < 0) {
            quack_wins++;
            outcome = "LOSS";
        } else {
            ties++;
            outcome = "TIE";
        }

        std::cout << std::left << std::setw(6) << (g + 1)
                  << std::setw(12) << (res.first_player == 0 ? "Tessera" : "Quackle")
                  << std::setw(10) << res.tessera_score
                  << std::setw(10) << res.quackle_score
                  << std::setw(10) << ((spread >= 0 ? "+" : "") + std::to_string(spread))
                  << std::setw(10) << res.tessera_bingos
                  << std::setw(10) << res.quackle_bingos
                  << std::setw(10) << outcome
                  << std::fixed << std::setprecision(1) << res.game_seconds << "s" << std::endl;
        std::cout.flush();
    }

    std::cout << "================================================================================" << std::endl;
    std::cout << "                               FINAL TOURNAMENT RESULTS                         " << std::endl;
    std::cout << "================================================================================" << std::endl;

    double win_rate = (double)(tess_wins + 0.5 * ties) / total_games * 100.0;
    double avg_tess_score = (double)total_tess_score / total_games;
    double avg_quack_score = (double)total_quack_score / total_games;
    double avg_spread = (double)(total_tess_score - total_quack_score) / total_games;
    double avg_tess_bingos = (double)total_tess_bingos / total_games;
    double avg_quack_bingos = (double)total_quack_bingos / total_games;

    double opp_base_elo = (opponent == "speedy") ? 1930.0 : 2060.0;
    double effective_wins = tess_wins + 0.5 * ties;
    double effective_losses = quack_wins + 0.5 * ties;
    if (effective_losses == 0) effective_losses = 0.5;
    if (effective_wins == 0) effective_wins = 0.5;
    double delta_elo = 400.0 * std::log10(effective_wins / effective_losses);
    double estimated_elo = opp_base_elo + delta_elo;

    std::cout << "  Record:               " << tess_wins << " Wins - " << quack_wins << " Losses - " << ties << " Ties" << std::endl;
    std::cout << "  Win Percentage:       " << std::fixed << std::setprecision(1) << win_rate << "%" << std::endl;
    std::cout << "  Average Scores:       Tessera " << std::setprecision(1) << avg_tess_score 
              << " - " << avg_quack_score << " Quackle" << std::endl;
    std::cout << "  Average Spread:       " << (avg_spread >= 0 ? "+" : "") << std::setprecision(1) << avg_spread << " pts/game" << std::endl;
    std::cout << "  Bingos / Game:        Tessera " << std::setprecision(2) << avg_tess_bingos 
              << " vs. Quackle " << avg_quack_bingos << std::endl;
    std::cout << "  Opponent Rating:      " << std::setprecision(0) << opp_base_elo << " Elo (" 
              << (opponent == "speedy" ? "SpeedyPlayer" : "SmartBogowin") << ")" << std::endl;
    std::cout << "  Delta Elo:            " << (delta_elo >= 0 ? "+" : "") << std::setprecision(1) << delta_elo << " Elo" << std::endl;
    std::cout << "  ESTIMATED ENGINE ELO: " << std::setprecision(0) << estimated_elo << " Elo" << std::endl;
    std::cout << "  Total Elapsed Time:   " << std::setprecision(1) << total_match_time << " s ("
              << (total_match_time / total_games) << " s/game)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    tessera_destroy_engine(engine);
    return 0;
}
