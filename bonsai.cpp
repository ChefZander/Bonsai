#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <bit>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "include/chess.hpp"

using namespace chess;

struct Node
{
    Move action;
    int firstchild; int num_children;
    int visits = 0;
    float value = 0;
};
struct SearchResult {
    Move bestMove;
    float confidence;

    // policy target
    std::vector<std::pair<Move, int>> policy;
};

Board board;
std::vector<Node> tree;

bool debug = false;
bool datagenActive = false;

int approxnps = 60000;

#include "net/net5.hpp"

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// smallnet
float evaluate_network(std::vector<float> features_768) {
    int32_t hidden_outputs[16] = {0};

    for (int i = 0; i < 16; ++i) {
        int32_t accumulation = HIDDEN_BIASES[i]; 
        for (int j = 0; j < 768; ++j) {
            if (features_768[j] == 1) { 
                accumulation += HIDDEN_WEIGHTS[i][j];
            }
        }
        hidden_outputs[i] = std::max(0, accumulation);
    }

    int32_t output_accumulation = OUTPUT_BIAS;
    for (int i = 0; i < 16; ++i) {
        output_accumulation += hidden_outputs[i] * OUTPUT_WEIGHTS[i];
    }

    float raw_output_float = (float)output_accumulation / SCALE_FACTOR_SQ;
    return sigmoid(raw_output_float);
}

// bignet
/*float evaluate_network(const std::vector<float>& features_768) {
    // Intermediate layer output buffers
    int64_t hl0_outputs[256] = {0};
    int64_t hl1_outputs[256] = {0};
    int64_t hl2_outputs[256] = {0};

    // --- LAYER 0: Input (768) -> HL0 (256) ---
    // Output Scale: SCALE_FACTOR^1
    for (int i = 0; i < 256; ++i) {
        int64_t accumulation = HL0_BIASES[i]; 
        for (int j = 0; j < 768; ++j) {
            if (features_768[j] == 1.0f) { 
                accumulation += HL0_WEIGHTS[i][j];
            }
        }
        hl0_outputs[i] = std::max(static_cast<int64_t>(0), accumulation); // ReLU
    }

    // --- LAYER 1: HL0 (256) -> HL1 (256) ---
    // Output Scale: SCALE_FACTOR^2
    for (int i = 0; i < 256; ++i) {
        int64_t accumulation = HL1_BIASES[i];
        for (int j = 0; j < 256; ++j) {
            accumulation += hl0_outputs[j] * HL1_WEIGHTS[i][j];
        }
        hl1_outputs[i] = std::max(static_cast<int64_t>(0), accumulation); // ReLU
    }

    // --- LAYER 2: HL1 (256) -> HL2 (256) ---
    // Output Scale: SCALE_FACTOR^3
    for (int i = 0; i < 256; ++i) {
        int64_t accumulation = HL2_BIASES[i];
        for (int j = 0; j < 256; ++j) {
            accumulation += hl1_outputs[j] * HL2_WEIGHTS[i][j];
        }
        hl2_outputs[i] = std::max(static_cast<int64_t>(0), accumulation); // ReLU
    }

    // --- LAYER 3: HL2 (256) -> Output (1) ---
    // Output Scale: SCALE_FACTOR^4
    int64_t output_accumulation = OUTPUT_BIAS;
    for (int i = 0; i < 256; ++i) {
        output_accumulation += hl2_outputs[i] * OUTPUT_WEIGHTS[i];
    }

    // --- DESCALING & ACTIVATION ---
    // Calculate final scale divisor: SCALE_FACTOR^4
    double final_scale = static_cast<double>(SCALE_FACTOR) * SCALE_FACTOR * SCALE_FACTOR * SCALE_FACTOR;
    
    // Convert back to floating point space
    float raw_output_float = static_cast<float>(static_cast<double>(output_accumulation) / final_scale);
    
    return sigmoid(raw_output_float);
}*/

// sillynet
/*float evaluate_network(const std::vector<float>& features_768) {
    // 1. Create a zero-padded input buffer [12 channels][10 rows][10 cols]
    // This removes the need for out-of-bounds branching inside the hot conv loops.
    int8_t padded_input[12][10][10] = {{{0}}};
    
    // Copy and cast features to integer once to avoid slow float->int instructions later
    for (int c = 0; c < 12; ++c) {
        int channel_offset = c * 64;
        for (int r = 0; r < 8; ++r) {
            int row_offset = channel_offset + (r * 8);
            for (int col = 0; col < 8; ++col) {
                padded_input[c][r + 1][col + 1] = static_cast<int8_t>(features_768[row_offset + col]);
            }
        }
    }

    int32_t output_accumulation = FC_BIAS;

    // 2. Perform Convolution, ReLU, and FC Accumulation in a tightly fused block
    // Loops strictly track the sequential layout of weights for maximum cache locality
    for (int r = 1; r <= 8; ++r) {
        for (int col = 1; col <= 8; ++col) {
            int32_t conv_out = CONV_BIAS;

            for (int c = 0; c < 12; ++c) {
                // Completely unrollable 3x3 kernel spatial iteration
                for (int dr = 0; dr < 3; ++dr) {
                    for (int dc = 0; dc < 3; ++dc) {
                        conv_out += padded_input[c][r + dr - 1][col + dc - 1] * CONV_WEIGHTS[c][dr][dc];
                    }
                }
            }

            // Fused ReLU Activation & FC layer mapping
            if (conv_out > 0) {
                int fc_index = (r - 1) * 8 + (col - 1);
                output_accumulation += conv_out * FC_WEIGHTS[fc_index];
            }
        }
    }

    // 3. De-quantize and scale back down to a float representation for the final step
    float raw_output_float = static_cast<float>(output_accumulation) / SCALE_FACTOR_SQ;
    return sigmoid(raw_output_float);
}*/

// this sucks, need to get rid of this
std::vector<float> fen_to_768(const std::string& fen) {
    std::vector<float> features(768, 0.0f);
    
    chess::Board board(fen);
    chess::Color side_to_move = board.sideToMove();
    
    for (int i = 0; i < 64; ++i) {
        chess::Square sq(i);
        chess::Piece piece = board.at(sq);
        
        if (piece == chess::Piece::NONE) {
            continue;
        }
        
        int type_idx = 0;
        switch (piece.type()) {
            case PieceType(PieceType::PAWN):   type_idx = 0; break;
            case PieceType(PieceType::KNIGHT): type_idx = 1; break;
            case PieceType(PieceType::BISHOP): type_idx = 2; break;
            case PieceType(PieceType::ROOK):   type_idx = 3; break;
            case PieceType(PieceType::QUEEN):  type_idx = 4; break;
            case PieceType(PieceType::KING):   type_idx = 5; break;
            default: continue;
        }
        
        int piece_idx = type_idx;
        if (piece.color() != side_to_move) {
            piece_idx += 6;
        }
        
        int chess_rank = static_cast<int>(sq.rank());
        int chess_file = static_cast<int>(sq.file());
        
        int python_square = (7 - chess_rank) * 8 + chess_file;
        if (side_to_move == chess::Color::BLACK) {
            python_square ^= 56; 
        }
        
        features[piece_idx * 64 + python_square] = 1.0f;
    }
    
    return features;
}

// obsolete?
bool isNodeTerminal() {
    return board.isGameOver() != std::pair<GameResultReason, GameResult>(GameResultReason::NONE, GameResult::NONE);
}

bool isNodeFullyExpanded(int node) {
    return tree[node].firstchild != 0;
}

void expandNode(int parent) {
    Movelist moves;
    movegen::legalmoves(moves, board);

    // appending at the end yields (idx + 1) as next index, which is size
    tree[parent].firstchild = tree.size();
    tree[parent].num_children = moves.size();

    for(Move m : moves) {
        Node child = Node();
        child.action = m;
        tree.push_back(child);
    }
}

const float C_PUCT = 1.41f;

// https://github.com/TomaszJaworski777/cpu-mcts-tutorial
float PUCT(int node, int parent) {
    float q;

    if (tree[node].visits == 0) {
        q = 0.5f;
    } else {
        q = tree[node].value / tree[node].visits;
    }

    float N_parent = static_cast<float>(tree[parent].visits);
    N_parent = std::max(N_parent, 1.0f);

    float N_child = static_cast<float>(tree[node].visits);
    float p = 1.0f / static_cast<float>(tree[parent].num_children);

    float expl_factor = std::sqrt(N_parent) / (N_child + 1.0f);

    float score = q + C_PUCT * p * expl_factor;

    return score;
}

int selectBestChild(int parent) {
    int bestChild = tree[parent].firstchild;
    float bestChildPUCT = -MAXFLOAT; // no MINFLOAT?

    for(int i = 0; i < tree[parent].num_children; i++) {
        float currentChildPUCT = PUCT(tree[parent].firstchild + i, parent);

        if(currentChildPUCT > bestChildPUCT) {
            bestChild = tree[parent].firstchild + i;
            bestChildPUCT = currentChildPUCT;
        }
    }

    return bestChild;
}

void backpropagateResult(std::vector<int> line, float value) {
    for(int i = line.size() - 1; i >= -1; i--) {
        tree[line[i]].visits++;
        tree[line[i]].value += value;
        value = 1 - value;
    }
}

// theoretically unused but leaving it here for static evaluation
int material(const chess::Board& b) {
    auto count = [](auto bb) { return std::popcount(bb.getBits()); };   

    return
        100 * (count(b.us(chess::Color::WHITE) & b.pieces(chess::PieceType::PAWN))
             - count(b.us(chess::Color::BLACK) & b.pieces(chess::PieceType::PAWN))) +
        320 * (count(b.us(chess::Color::WHITE) & b.pieces(chess::PieceType::KNIGHT))
             - count(b.us(chess::Color::BLACK) & b.pieces(chess::PieceType::KNIGHT))) +
        330 * (count(b.us(chess::Color::WHITE) & b.pieces(chess::PieceType::BISHOP))
             - count(b.us(chess::Color::BLACK) & b.pieces(chess::PieceType::BISHOP))) +
        500 * (count(b.us(chess::Color::WHITE) & b.pieces(chess::PieceType::ROOK))
             - count(b.us(chess::Color::BLACK) & b.pieces(chess::PieceType::ROOK))) +
        900 * (count(b.us(chess::Color::WHITE) & b.pieces(chess::PieceType::QUEEN))
             - count(b.us(chess::Color::BLACK) & b.pieces(chess::PieceType::QUEEN)));
}

SearchResult monteCarloSearch(Board state, int iterationsMax, int timeMax) {
    if(iterationsMax == 0) iterationsMax = INT32_MAX;

    tree.clear(); // never forget

    Node root = Node();
    tree.push_back(root); // root node is always index 0
    expandNode(0); // expand root for the first iteration

    auto startTime = std::chrono::steady_clock::now();

    long long plyCurrent = 0; // current depth at this point in the search
    long long plyTotal = 0; // total amount of depth reached (basically meaningless outside of the average calculation)
    int plyMax = 0; // deepest ply reached in search so far

    int iterationsCompleted = 0;

    for(; iterationsCompleted < iterationsMax; iterationsCompleted++) {
        std::vector<int> line; // saves the indecies of the nodes in the tree taken as moves on the board

        // index of the current node to look at
        int current = 0;

        // descend down the tree until finding a leaf
        while (isNodeFullyExpanded(current))
        {
            current = selectBestChild(current);
            board.makeMove(tree[current].action);
            plyCurrent++;
            line.push_back(current);
        }

        // expand the leaf and select again from it's children
        if(!isNodeTerminal() && !isNodeFullyExpanded(current)) {
            expandNode(current);
            current = selectBestChild(current);
            board.makeMove(tree[current].action);
            plyCurrent++;
            line.push_back(current);
        }

        // update plyMax
        if (plyCurrent > plyMax) plyMax = plyCurrent;
        plyTotal += plyCurrent;
        plyCurrent = 0;

        // check if we can skip evaluation
        GameResult r = board.isGameOver().second;
        float value = 0;

        switch(r) {
            case GameResult::WIN:
                value = 1;
                break;
            case GameResult::DRAW:
                value = 0.5;
                break;
            case GameResult::LOSE:
                value = 0;
                break;
            case GameResult::NONE:
                // game not done yet, have to evaluate
                value = evaluate_network(fen_to_768(board.getFen()));
                value *= 0.9; // so that the network can't drown a win signal

                //value = 1 / (1 + std::exp(-material(board) / 400));
                //if(board.sideToMove() == Color::BLACK) value = 1 - value;
        }

        backpropagateResult(line, 1 - value);

        // get the board back into root state for the next iteration
        // could potentially optimize here, like only going back a few moves if you already know you're taking the same starting moves 
        for(int i = line.size() - 1; i >= 0; i--) {
            board.unmakeMove(tree[line[i]].action);
        }

        // check if we have time
        auto currentTime = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsedSeconds = currentTime - startTime;
        double safeElapsed = std::max(1e-9, elapsedSeconds.count());
        if(timeMax < (safeElapsed*1000) && timeMax != 0) {
            break;
        }

        if(!datagenActive) {
            if(iterationsCompleted % (approxnps/4) == 0 && iterationsCompleted > 0) {
                // build the PV from the tree
                std::vector<int> pvLine;
                int pvc = 0;
                while(isNodeFullyExpanded(pvc)) {
                    int bestChild = 0;
                    float bestChildVisits = -1;

                    for(int i = 0; i < tree[pvc].num_children; i++) {
                        int currentNode = tree[pvc].firstchild + i;
                        if(tree[currentNode].visits > bestChildVisits) {
                            bestChild = currentNode;
                            bestChildVisits = tree[currentNode].visits;
                        }
                    }

                    pvLine.push_back(bestChild);
                    pvc = bestChild;
                }
                
                int simPerSec = static_cast<int>(iterationsCompleted / safeElapsed);

                /*int bestChild = pvLine[0];
                int bestChildVisits = tree[pvLine[0]].visits;
                int bestChildValue = tree[pvLine[0]].value;*/
                float y = floor(((1 - (static_cast<float>(tree[0].value) / tree[0].visits)) - 0.5) * 100 * 16);
                //int eval = static_cast<int>(400.0f * std::log(y / (1.0f - y)));

                std::cout << "info depth " << (plyTotal/iterationsCompleted) << " seldepth " << plyMax << " nodes " << iterationsCompleted << " nps " << simPerSec << " time " << static_cast<int>(elapsedSeconds.count()*1000) << " score cp " << y << " pv ";
                for(int node : pvLine) {
                    std::cout << uci::moveToUci(tree[node].action) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    // calculate PV once again
    std::vector<int> pvLine;
    int pvc = 0;
    while(isNodeFullyExpanded(pvc)) {
        int bestChild = 0;
        float bestChildVisits = -1;

        for(int i = 0; i < tree[pvc].num_children; i++) {
            int currentNode = tree[pvc].firstchild + i;
            if(tree[currentNode].visits > bestChildVisits) {
                bestChild = currentNode;
                bestChildVisits = tree[currentNode].visits;
            }
        }

        pvLine.push_back(bestChild);
        pvc = bestChild;
    }
    
    int bestChild = pvLine[0];
    int bestChildVisits = tree[pvLine[0]].visits;
    int bestChildValue = tree[pvLine[0]].value;

    auto currentTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsedSeconds = currentTime - startTime;
    double safeElapsed = std::max(1e-9, elapsedSeconds.count());
    int simPerSec = static_cast<int>(iterationsCompleted / safeElapsed);

    float y = floor(((1 - (static_cast<float>(tree[0].value) / tree[0].visits)) - 0.5) * 100 * 16);

    //int eval = static_cast<int>(400.0f * std::log(y / (1.0f - y)));

    // uci info printouts
    if(!datagenActive) {
        std::cout << "info depth " << (plyTotal/iterationsCompleted) << " seldepth " << plyMax << " nodes " << iterationsCompleted << " nps " << simPerSec << " time " << static_cast<int>(elapsedSeconds.count()*1000) << " score cp " << y << " pv ";
        for(int node : pvLine) {
            std::cout << uci::moveToUci(tree[node].action) << " ";
        }
        std::cout << std::endl;
    }
    
    // debugging output for move probabilities at root
    if(debug && !datagenActive) {
        // Print visit counts relative to the maximum
        const int BAR_WIDTH = 40;

        std::cout << "\nMove Statistics:\n";
        std::cout << "-------------------------------\n";

        for (int i = 0; i < tree[0].num_children; i++) {
            Node child = tree[tree[0].firstchild + i];

            float ratio = static_cast<float>(child.visits) / iterationsCompleted;

            int filled = static_cast<int>(ratio * BAR_WIDTH);

            std::cout << " ";

            std::cout << "[";
            for (int j = 0; j < BAR_WIDTH; j++) {
                if (j < filled)
                    std::cout << "#";
                else
                    std::cout << " ";
            }
            std::cout << "] " << uci::moveToUci(child.action) << " ";

            std::cout << child.visits << " ("
                    << std::fixed
                    << ratio * 100.0f << "%" << ") q=" << (child.value / child.visits) << " puct=" << PUCT(tree[0].firstchild + i, 0) << "\n";
            //PUCTroot(child);
        }
    }

    SearchResult result;

    result.bestMove = tree[bestChild].action;
    result.confidence = 1 - (static_cast<float>(tree[0].value) / tree[0].visits);

    for (int i = 0; i < tree[0].num_children; i++) {
        Node &child = tree[tree[0].firstchild + i];

        result.policy.push_back({
            child.action,
            child.visits
        });
    }

    return result;
}


struct DatagenPosition {
    std::string fen;
    std::vector<std::pair<Move, int>> policy;
    Color sideToMove;
    float confidence;
};

void writeGameToCSV(const std::string& filename,
                    const std::vector<DatagenPosition>& game,
                    GameResult result,
                    Color winner)
{
    std::ofstream file(filename, std::ios::app);

    for (const auto& position : game)
    {
        int target;

        if (result == GameResult::DRAW)
        {
            target = 0;
        }
        else
        {
            target = (position.sideToMove == winner) ? 1 : -1;
        }

        file << position.fen << ",";

        // uncomment to save policy output as well, but this uses ~3x more data
        /*bool first = true;
        for (const auto& p : position.policy)
        {
            if (!first)
                file << ";";

            first = false;

            file << uci::moveToUci(p.first) // Move move
                 << ":"
                 << p.second; // int visits
        }*/

        //file << "," << target << "\n";
        file << position.confidence << "\n";
    }
}

void datagen() {
    datagenActive = true;
    int i = 0;
    while (true) {
        // make a random position
        board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        int numMoves = 5 + rand() % 5;

        for (int j = 0; j < numMoves; j++) {
            Movelist moves;
            movegen::legalmoves(moves, board);

            if (moves.empty()) {
                break;
            }

            int index = rand() % moves.size();
            board.makeMove(moves[index]);
        }
        std::string startingFen = board.getFen();

        // loop monte carlo until game ends
        std::vector<DatagenPosition> game;
        int ply = 0;
        while (true) {
            Movelist moves;
            movegen::legalmoves(moves, board);

            if (board.isGameOver().second != GameResult::NONE)
                break;

            if (rand() % 10 == 0) {
                int index = rand() % moves.size();
                board.makeMove(moves[index]);
            }
            else {
                SearchResult search = monteCarloSearch(board, 501, 0);

                DatagenPosition pos;
                pos.fen = board.getFen();
                pos.policy = search.policy;
                pos.sideToMove = board.sideToMove();
                pos.confidence = search.confidence;

                // TODO: only save quiet positions?
                game.push_back(pos);

                board.makeMove(search.bestMove);
            }
            ply++;
        }

        // save that shish to a file
        auto [over, result] = board.isGameOver();

        Color winner = Color::NONE; // stays none if draw

        if (result == GameResult::LOSE)
        {
            // sideToMove is checkmated, so opposite side won
            winner = (board.sideToMove() == Color::WHITE)
                ? Color::BLACK
                : Color::WHITE;
        }

        std::cout << "Game " << i << ": " << startingFen << " plies: " << ply << " winner: " << winner << std::endl;

        writeGameToCSV("data/selfplay_2.csv", game, result, winner);
        game.clear();
        i++;
    }
}

void handlePosition(std::istringstream& ss) {
    std::string token;
    ss >> token; // Should be "startpos" or "fen"

    if (token == "startpos") {
        board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    } else if (token == "fen") {
        std::string fen_string;
        // Read the FEN string- can have spaces
        while (ss >> token && token != "moves") {
            fen_string += token + " ";
        }
        if (!fen_string.empty()) {fen_string.pop_back();}
        board.setFen(fen_string);

        if (token == "moves") {
            while (ss >> token) {
                //std::cout << token << std::endl;
                chess::Move move = uci::uciToMove(board, token);
                board.makeMove(move);
            }
        }
        return;
    }

    ss >> token; // Should be "moves"
    if (token == "moves") {
        while (ss >> token) {
            //std::cout << token << std::endl;
            chess::Move move = uci::uciToMove(board, token);
            board.makeMove(move);
        }
    }
}

int main(int argc, char* argv[]) {

    std::string line;
    board = Board();
    board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "uci") {
            std::cout << "id name Bonsai" << std::endl;
            std::cout << "id author Zander" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (command == "ucinewgame") {
            board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        } else if (command == "position") {
            handlePosition(iss);
        } else if (command == "eval") {
            std::cout << "Evaluation for this position: " << material(board) << std::endl;
            std::cout << "Evaluation for this position: " << evaluate_network(fen_to_768(board.getFen())) << std::endl;
        } else if (command == "go") {
            // Set your default search limits if the GUI just sends "go"
            int nodes = 3000000;
            int movetime = 30000; 

            std::string token;
            while (iss >> token) {
                if (token == "nodes") {
                    iss >> nodes;
                } else if (token == "movetime") {
                    iss >> movetime;
                }
            }

            Move bestMove = monteCarloSearch(board, nodes, movetime).bestMove;
            std::cout << "bestmove " << uci::moveToUci(bestMove) << std::endl;
        } else if (command == "dg") {
            datagen();
        } else if (command == "quit") {
            break;
        }
    }

    return 0;
}