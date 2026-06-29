#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <bit>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <queue>
#include "include/chess.hpp"

using namespace chess;

struct Node
{
    Move action;
    int firstchild; int num_children;
    int visits = 0;
    float value = 0;
    bool captureMove;
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

const int DEFAULT_HASH_MIB = 128;
int hashMib = DEFAULT_HASH_MIB;
size_t hashLimitNodes = static_cast<size_t>(DEFAULT_HASH_MIB) * 1024 * 1024 / sizeof(Node);

inline size_t computeHashLimitNodes(int mib) {
    return static_cast<size_t>(mib) * 1024 * 1024 / sizeof(Node);
}

// Fraction of total nodes pruneTree() retains per call (~half).
const float PRUNE_KEEP_FRACTION = 0.5f;

#include "net/net7-2.hpp"

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// SmallNet inference using accumulators
// Instead of building a full 768-feature vector and scanning all 768 x 16 weights,
// we initialize 16 hidden accumulators with biases and only add contributions
// for the ~30 actually-set features (one per piece on the board).
// This reduces the hot loop from O(768*16) to O(numPieces*16).
inline float evaluate_network(const chess::Board& b) {
    int32_t acc[16];
    for (int i = 0; i < 16; ++i) {
        acc[i] = HIDDEN_BIASES[i];
    }

    chess::Color stm = b.sideToMove();
    bool isBlack = (stm == chess::Color::BLACK);

    // Precompute rank/file lookup to avoid constructing chess::Square twice
    for (int sq = 0; sq < 64; ++sq) {
        chess::Piece piece = b.at(chess::Square(sq));
        if (piece == chess::Piece::NONE) continue;

        int type_idx;
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
        if (piece.color() != stm) {
            piece_idx += 6;
        }

        // sq is 0-63: rank = sq / 8, file = sq % 8  (algebraic: a1=0, h1=7, a8=56, h8=63)
        int rank = sq >> 3;      // sq / 8
        int file = sq & 7;       // sq % 8
        int python_square = (7 - rank) * 8 + file;
        if (isBlack) {
            python_square ^= 56;
        }

        int feature_idx = piece_idx * 64 + python_square;

        // Accumulate contribution of this single active feature into all 16 hidden neurons
        for (int h = 0; h < 16; ++h) {
            acc[h] += HIDDEN_WEIGHTS[h][feature_idx];
        }
    }

    // ReLU on hidden outputs, then accumulate into final output
    int32_t output_accumulation = OUTPUT_BIAS;
    for (int i = 0; i < 16; ++i) {
        int32_t h = acc[i];
        if (h > 0) {
            output_accumulation += h * OUTPUT_WEIGHTS[i];
        }
    }

    float raw_output_float = static_cast<float>(output_accumulation) / SCALE_FACTOR_SQ;
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

// obsolete?
inline bool isNodeTerminal(GameResult result) {
    return result != GameResult::NONE;
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
        child.captureMove = (board.at(m.to()) != Piece::NONE);
        tree.push_back(child);
    }
}

const float C_PUCT = 1.41f;

// Fast sqrt approximation using SSE-friendly integer math
// Good enough for MCTS selection; avoids expensive std::sqrt call in hot path
inline float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    // Use hardware sqrt hint; compilers will emit sqrtss on x86
    float r = sqrtf(x);
    return r;
}

// https://github.com/TomaszJaworski777/cpu-mcts-tutorial
inline float PUCT(int node, int parent) {
    const Node& child = tree[node];
    const Node& parent_node = tree[parent];

    float q = (child.visits == 0) ? 0.5f : (child.value / child.visits);

    float N_parent = static_cast<float>(parent_node.visits);
    if (N_parent < 1.0f) N_parent = 1.0f;

    float p = fast_sqrt(N_parent) / (static_cast<float>(child.visits) + 1.0f);

    // increase policy for capture moves slightly to encourage exploration first
    if(tree[node].captureMove) {
        p += 0.1f;
    }

    return q + C_PUCT * p / static_cast<float>(parent_node.num_children);
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

void backpropagateResult(const std::vector<int>& line, float value) {
    // Backpropagate through the path from leaf to child of root
    for (int i = static_cast<int>(line.size()) - 1; i >= 0; i--) {
        tree[line[i]].visits++;
        tree[line[i]].value += value;
        value = 1 - value;
    }
    // Update root node
    tree[0].visits++;
    tree[0].value += value;
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

// Helper: build PV line (most-visited child path) from root into `out`
void buildPV(std::vector<int>& out) {
    out.clear();
    int cur = 0;
    while (isNodeFullyExpanded(cur)) {
        int bestChild = 0;
        int bestVisits = -1;
        for (int i = 0; i < tree[cur].num_children; i++) {
            int idx = tree[cur].firstchild + i;
            if (tree[idx].visits > bestVisits) {
                bestChild = idx;
                bestVisits = tree[idx].visits;
            }
        }
        out.push_back(bestChild);
        cur = bestChild;
    }
}

// Helper: print UCI info line
void printInfo(long long plyTotal, int plyMax, int iterationsCompleted,
               double safeElapsed, const std::vector<int>& pvLine) {
    int simPerSec = static_cast<int>(iterationsCompleted / safeElapsed);
    int hashfull = 0;
    if (hashLimitNodes > 0) {
        size_t used = tree.size();
        if (used > hashLimitNodes) used = hashLimitNodes;
        hashfull = static_cast<int>((used * 1000) / hashLimitNodes);
    }
    float y = floor(((1.0f - (static_cast<float>(tree[0].value) / tree[0].visits)) - 0.5f) * 100.0f * 20.0f);

    std::cout << "info depth " << (plyTotal / iterationsCompleted)
              << " seldepth " << plyMax
              << " nodes " << iterationsCompleted
              << " nps " << simPerSec
              << " hashfull " << hashfull
              << " time " << static_cast<int>(safeElapsed * 1000)
              << " score cp " << y
              << " pv ";
    for (int node : pvLine) {
        std::cout << uci::moveToUci(tree[node].action) << " ";
    }
    std::cout << std::endl;
}

void pruneTree() {
    const size_t n = tree.size();
    if (n < 1024) return; // not worth compacting tiny trees

    size_t keepTarget = static_cast<size_t>(n * PRUNE_KEEP_FRACTION);
    if (keepTarget < 1) keepTarget = 1;
    if (keepTarget > n) keepTarget = n;

    // --- Pass A: mark the top-`keepTarget` nodes by (visits desc, index asc) ---
    // That set is ancestor-closed (see above), so it is a connected subtree
    // rooted at node 0 and compacts cleanly.
    std::vector<bool> kept(n, false);
    {
        std::vector<std::pair<int,int>> rank; // (visits, index)
        rank.reserve(n);
        for (size_t i = 0; i < n; i++)
            rank.push_back({tree[i].visits, static_cast<int>(i)});
        std::partial_sort(rank.begin(), rank.begin() + keepTarget, rank.end(),
                          [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                              if (a.first != b.first) return a.first > b.first;
                              return a.second < b.second;
                          });
        for (size_t i = 0; i < keepTarget; i++) kept[rank[i].second] = true;
        kept[0] = true; // root always retained (defensive)
    } // `rank` freed before compaction to keep peak memory down

    // --- Pass B: emit the kept subtree in BFS order ---
    // BFS lets us write each kept node's kept children as one contiguous block
    // (same layout expandNode() produces), so firstchild + num_children stay
    // valid. Subtrees may land anywhere in the array -- that's fine, they are
    // reached through firstchild, exactly like the organically grown tree.
    size_t keptCount = 0;
    for (size_t i = 0; i < n; i++) if (kept[i]) keptCount++;

    std::vector<Node> newTree;
    newTree.reserve(keptCount);
    std::vector<int> oldToNew(n, -1);
    std::vector<int> keptKids; // reused
    std::queue<int> bfs;

    oldToNew[0] = 0;
    newTree.push_back(tree[0]);
    bfs.push(0);

    while (!bfs.empty()) {
        int p = bfs.front(); bfs.pop();
        int pNew = oldToNew[p];
        const Node& pnode = tree[p];

        keptKids.clear();
        if (pnode.firstchild != 0) {
            for (int i = 0; i < pnode.num_children; i++) {
                int ci = pnode.firstchild + i;
                if (kept[ci]) keptKids.push_back(ci);
            }
        }

        if (keptKids.empty()) {
            // No retained children: this node becomes a re-expandable leaf.
            newTree[pNew].firstchild = 0;
            newTree[pNew].num_children = 0;
            continue;
        }

        int childBlockStart = static_cast<int>(newTree.size());
        newTree[pNew].firstchild = childBlockStart;
        newTree[pNew].num_children = static_cast<int>(keptKids.size());
        for (int ci : keptKids) {
            oldToNew[ci] = static_cast<int>(newTree.size());
            newTree.push_back(tree[ci]); // firstchild fixed when ci is processed
            bfs.push(ci);
        }
    }

    if (debug && !datagenActive) {
        std::cerr << "info string prune " << n << " -> " << newTree.size()
                  << " nodes (hash limit " << hashLimitNodes << ")" << std::endl;
    }

    tree = std::move(newTree);
}

SearchResult monteCarloSearch(int iterationsMax, int timeMax) {
    if (iterationsMax == 0) iterationsMax = INT32_MAX;

    tree.clear();

    Node root = Node();
    tree.push_back(root);
    expandNode(0);

    auto startTime = std::chrono::steady_clock::now();

    long long plyCurrent = 0;
    long long plyTotal = 0;
    int plyMax = 0;
    int iterationsCompleted = 0;

    // Reusable buffers to avoid per-iteration allocations
    std::vector<int> line;
    line.reserve(64);
    std::vector<int> pvLine;
    pvLine.reserve(32);

    int infoInterval = approxnps / 4;
    int nextInfo = infoInterval;
    int nextTimeCheck = 2500;

    for (; iterationsCompleted < iterationsMax; iterationsCompleted++) {
        line.clear();
        int current = 0;

        // Selection: descend to a leaf
        while (isNodeFullyExpanded(current)) {
            current = selectBestChild(current);
            board.makeMove(tree[current].action);
            plyCurrent++;
            line.push_back(current);
        }

        // Expansion + one more selection step
        auto [over, r] = board.isGameOver();
        if (r == GameResult::NONE && !isNodeFullyExpanded(current)) {
            expandNode(current);
            current = selectBestChild(current);
            board.makeMove(tree[current].action);
            plyCurrent++;
            line.push_back(current);
            // Re-check game state after the extra move
            auto result2 = board.isGameOver();
            r = result2.second;
        }

        // Track depth stats
        if (plyCurrent > plyMax) plyMax = plyCurrent;
        plyTotal += plyCurrent;
        plyCurrent = 0;

        // Evaluation
        float value = 0;
        switch (r) {
            case GameResult::WIN:  value = 1.0f; break;
            case GameResult::DRAW: value = 0.5f; break;
            case GameResult::LOSE: value = 0.0f; break;
            case GameResult::NONE: value = evaluate_network(board) * 0.9f; break;
        }

        backpropagateResult(line, 1.0f - value);

        // Unmake moves to restore root position
        for (int i = static_cast<int>(line.size()) - 1; i >= 0; i--) {
            board.unmakeMove(tree[line[i]].action);
        }

        // compact
        if (tree.size() >= hashLimitNodes) {
            pruneTree();
        }

        // Periodic time check (not every iteration)
        if (timeMax != 0 && iterationsCompleted >= nextTimeCheck) {
            auto currentTime = std::chrono::steady_clock::now();
            double elapsed = std::max(1e-9, std::chrono::duration<double>(currentTime - startTime).count());
            if (timeMax < elapsed * 1000.0) break;
            nextTimeCheck += 2500;
        }

        // Periodic info print
        if (!datagenActive && iterationsCompleted >= nextInfo) {
            auto currentTime = std::chrono::steady_clock::now();
            double safeElapsed = std::max(1e-9, std::chrono::duration<double>(currentTime - startTime).count());
            buildPV(pvLine);
            printInfo(plyTotal, plyMax, iterationsCompleted, safeElapsed, pvLine);
            nextInfo += infoInterval;
        }
    }

    // Final PV and info
    buildPV(pvLine);

    auto currentTime = std::chrono::steady_clock::now();
    double safeElapsed = std::max(1e-9, std::chrono::duration<double>(currentTime - startTime).count());

    if (!datagenActive) {
        printInfo(plyTotal, plyMax, iterationsCompleted, safeElapsed, pvLine);
    }

    // Debug output
    if (debug && !datagenActive) {
        const int BAR_WIDTH = 40;
        std::cout << "\nMove Statistics:\n";
        std::cout << "-------------------------------\n";

        for (int i = 0; i < tree[0].num_children; i++) {
            const Node& child = tree[tree[0].firstchild + i];
            float ratio = static_cast<float>(child.visits) / iterationsCompleted;
            int filled = static_cast<int>(ratio * BAR_WIDTH);

            std::cout << " [";
            for (int j = 0; j < BAR_WIDTH; j++) {
                std::cout << (j < filled ? '#' : ' ');
            }
            std::cout << "] " << uci::moveToUci(child.action) << " "
                      << child.visits << " ("
                      << std::fixed << ratio * 100.0f << "%" << ") q="
                      << (child.visits > 0 ? child.value / child.visits : 0.0f)
                      << " puct=" << PUCT(tree[0].firstchild + i, 0) << "\n";
        }
    }

    int bestChild = pvLine.empty() ? tree[0].firstchild : pvLine[0];

    SearchResult result;
    result.bestMove = tree[bestChild].action;
    result.confidence = 1.0f - (static_cast<float>(tree[0].value) / tree[0].visits);

    result.policy.reserve(tree[0].num_children);
    for (int i = 0; i < tree[0].num_children; i++) {
        const Node& child = tree[tree[0].firstchild + i];
        result.policy.push_back({child.action, child.visits});
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
                SearchResult search = monteCarloSearch(2001, 0);

                DatagenPosition pos;
                pos.fen = board.getFen();
                pos.policy = search.policy;
                pos.sideToMove = board.sideToMove();
                pos.confidence = search.confidence;

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

        writeGameToCSV("data/selfplay_3.csv", game, result, winner);
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
            std::cout << "option name Hash type spin default 128 min 1 max 65536" << std::endl;
            std::cout << "uciok" << std::endl;
        } else if (command == "debug") {
            std::string arg;
            iss >> arg;
            if (arg == "on" || arg == "true" || arg == "1") {
                debug = true;
            } else if (arg == "off" || arg == "false" || arg == "0" || arg.empty()) {
                debug = false;
            }
        } else if (command == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (command == "ucinewgame") {
            board.setFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        } else if (command == "setoption") {
            std::string token, name;
            iss >> token;   // "name"
            iss >> name;
            iss >> token;   // "value"
            if (name == "Hash") {
                iss >> hashMib;
                if (hashMib < 1) hashMib = 1;
                hashLimitNodes = computeHashLimitNodes(hashMib);
            }
        } else if (command == "position") {
            handlePosition(iss);
        } else if (command == "eval") {
            std::cout << "Evaluation for this position: " << material(board) << std::endl;
            std::cout << "Evaluation for this position: " << evaluate_network(board) << std::endl;
        } else if (command == "go") {
            // Set your default search limits if the GUI just sends "go"
            int nodes = 0;
            int movetime = 0; 

            std::string token;
            while (iss >> token) {
                if (token == "nodes") {
                    iss >> nodes;
                } else if (token == "movetime") {
                    iss >> movetime;
                }
            }

            Move bestMove = monteCarloSearch(nodes, movetime).bestMove;
            std::cout << "bestmove " << uci::moveToUci(bestMove) << std::endl;
        } else if (command == "dg") {
            datagen();
        } else if (command == "quit") {
            break;
        }
    }

    return 0;
}