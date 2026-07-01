#include <stdio.h>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <bit>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <queue>
#include <immintrin.h>
#include <random>
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

#include "net/64hl1.hpp"

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float evaluate_network_32hl(const chess::Board& b) {
    int32_t acc[32];
    for (int i = 0; i < 32; ++i) {
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
        for (int h = 0; h < 32; ++h) {
            acc[h] += HIDDEN_WEIGHTS[h][feature_idx];
        }
    }

    // SCReLU
    constexpr int32_t CLAMP_LIMIT = 255; 
    int32_t output_accumulation = OUTPUT_BIAS;

    for (int i = 0; i < 32; ++i) {
        int32_t h = acc[i];
        
        // Branchless clamp to [0, CLAMP_LIMIT]
        h = (h < 0) ? 0 : ((h > CLAMP_LIMIT) ? CLAMP_LIMIT : h);
        
        // Square the clipped value and accumulate into output
        output_accumulation += h * h * OUTPUT_WEIGHTS[i];
    }

    float raw_output_float = static_cast<float>(output_accumulation) / SCALE_FACTOR_SQ;
    return sigmoid(raw_output_float);
}

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

inline float fast_sqrt_hint(float x) {
    if (x <= 0.0f) return 0.0f;
    #if defined(__SSE2__)
    __m128 v = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(v); // r ~ 1/sqrt(x)
    
    // Newton-Raphson refinement for sqrt(x):
    // result = 0.5 * (x * r) * (3.0 - x * r * r)
    __m128 x_r = _mm_mul_ss(v, r);
    __m128 x_r_r = _mm_mul_ss(x_r, r);
    __m128 three = _mm_set_ss(3.0f);
    __m128 half = _mm_set_ss(0.5f);
    
    __m128 term = _mm_sub_ss(three, x_r_r);
    __m128 refined = _mm_mul_ss(half, _mm_mul_ss(x_r, term));
    
    float result;
    _mm_store_ss(&result, refined);
    return result;
    #else
    return sqrtf(x);
    #endif
}

// https://github.com/TomaszJaworski777/cpu-mcts-tutorial
inline float PUCT(int node, int parent) {
    const Node& child = tree[node];
    const Node& parent_node = tree[parent];

    float q = (child.visits == 0) ? 0.5f : (child.value / child.visits);

    float N_parent = static_cast<float>(parent_node.visits);
    if (N_parent < 1.0f) N_parent = 1.0f;

    float p = fast_sqrt_hint(N_parent) / (static_cast<float>(child.visits) + 1.0f);

    return q + C_PUCT * p;
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
inline int material(const chess::Board& b) {
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
        std::cout << "info string prune " << n << " -> " << newTree.size()
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

    int infoInterval = 100000;
    int nextInfo = infoInterval;
    int nextTimeCheck = 1000;

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

            // removed * 0.9f, either tune or throw out, is probably bad for eval
            case GameResult::NONE: value = evaluate_network_64hl(board); break;

            // fallback
            //case GameResult::NONE: value = 1.0 / (1.0 + std::exp(-static_cast<double>(material(board)) / 400.0)); break;
        }

        backpropagateResult(line, 1.0f - value);

        // Unmake moves to restore root position
        for (int i = static_cast<int>(line.size()) - 1; i >= 0; i--) {
            board.unmakeMove(tree[line[i]].action);
        }

        // Periodic time check (not every iteration)
        if (timeMax != 0 && iterationsCompleted >= nextTimeCheck) {
            auto currentTime = std::chrono::steady_clock::now();
            double elapsed = std::max(1e-9, std::chrono::duration<double>(currentTime - startTime).count());
            if (timeMax < elapsed * 1000.0) break;
            nextTimeCheck += 1000;
        }

        // compact
        if (tree.size() >= hashLimitNodes) {
            pruneTree();
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

#pragma pack(push, 1)
struct PositionRecord {
    uint64_t bitboards[12]; // 96 bytes: 768-bit one-hot encoding
    float confidence;       // 4 bytes: MCTS search evaluation
};
#pragma pack(pop) // Total size: Exactly 100 bytes per position

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

        //file << "," << position.confidence << "\n";
        file << position.confidence << "\n";
    }
}

void writeGameToBinary(const std::string& filename,
                       const std::vector<DatagenPosition>& game,
                       const Color& winner)
{
    std::ofstream file(filename, std::ios::binary | std::ios::app);
    if (!file.is_open()) return;

    std::vector<PositionRecord> records;
    records.reserve(game.size());

    for (const auto& position : game)
    {
        PositionRecord record;

        float gameResult = 0.5f; // Default to a draw
        if (winner == Color::WHITE) {
            gameResult = (position.sideToMove == Color::WHITE) ? 1.0f : 0.0f;
        } else if (winner == Color::BLACK) {
            gameResult = (position.sideToMove == Color::BLACK) ? 1.0f : 0.0f;
        }
        
        // Apply lambda blend formula
        record.confidence = (0.7f * position.confidence) + (0.3f * gameResult);

        std::memset(record.bitboards, 0, sizeof(record.bitboards));

        bool isWhite = (position.sideToMove == Color::WHITE);
        int rank = 7;
        int fileIdx = 0;

        for (char c : position.fen)
        {
            if (c == ' ') break;
            
            if (c == '/') {
                rank--;
                fileIdx = 0;
            }
            else if (std::isdigit(c)) {
                fileIdx += c - '0';
            }
            else {
                int pIdx = -1;
                // Bake PIECE_MAP_W vs PIECE_MAP_B logic directly into the parser
                if (isWhite) {
                    switch (c) {
                        case 'P': pIdx = 0;  break; case 'N': pIdx = 1;  break;
                        case 'B': pIdx = 2;  break; case 'R': pIdx = 3;  break;
                        case 'Q': pIdx = 4;  break; case 'K': pIdx = 5;  break;
                        case 'p': pIdx = 6;  break; case 'n': pIdx = 7;  break;
                        case 'b': pIdx = 8;  break; case 'r': pIdx = 9;  break;
                        case 'q': pIdx = 10; break; case 'k': pIdx = 11; break;
                    }
                } else {
                    switch (c) {
                        case 'p': pIdx = 0;  break; case 'n': pIdx = 1;  break;
                        case 'b': pIdx = 2;  break; case 'r': pIdx = 3;  break;
                        case 'q': pIdx = 4;  break; case 'k': pIdx = 5;  break;
                        case 'P': pIdx = 6;  break; case 'N': pIdx = 7;  break;
                        case 'B': pIdx = 8;  break; case 'R': pIdx = 9;  break;
                        case 'Q': pIdx = 10; break; case 'K': pIdx = 11; break;
                    }
                }

                if (pIdx != -1) {
                    int square = rank * 8 + fileIdx;
                    // Flip the board vertically for Black's perspective (square ^ 56)
                    if (!isWhite) {
                        square ^= 56;
                    }
                    
                    record.bitboards[pIdx] |= (1ULL << square);
                    fileIdx++;
                }
            }
        }
        records.push_back(record);
    }

    file.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(PositionRecord));
}

void datagen() {
    datagenActive = true;
    int i = 0;
    
    // Set up a proper random engine for weighted sampling
    std::random_device rd;
    std::mt19937 gen(rd());

    while (true) {
        // Make a random opening position
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

        std::vector<DatagenPosition> game;
        int ply = 0;

        // --- START TIMER ---
        auto startTime = std::chrono::high_resolution_clock::now();

        while (true) {
            Movelist moves;
            movegen::legalmoves(moves, board);

            if (board.isGameOver().second != GameResult::NONE)
                break;

            // Run the deep MCTS search for every recorded position
            SearchResult search = monteCarloSearch(1501, 0);

            DatagenPosition pos;
            pos.fen = board.getFen();
            pos.policy = search.policy;
            pos.sideToMove = board.sideToMove();
            pos.confidence = search.confidence;
            game.push_back(pos);

            Move moveToPlay;

            // --- TEMPERATURE MECHANISM ---
            if (ply < 20) { 
                // T = 1: Weighted random choice based strictly on visit counts
                int totalVisits = 0;
                for (const auto& p : search.policy) {
                    totalVisits += p.second;
                }

                if (totalVisits > 0) {
                    std::uniform_int_distribution<> dis(0, totalVisits - 1);
                    int target = dis(gen);
                    int currentSum = 0;

                    for (const auto& p : search.policy) {
                        currentSum += p.second;
                        if (currentSum > target) {
                            moveToPlay = p.first;
                            break;
                        }
                    }
                } else {
                    moveToPlay = search.bestMove; // Fallback safety
                }
            } 
            else { 
                // T = 0: Exploit mode. Play the absolute best move found by the search
                moveToPlay = search.bestMove;
            }

            board.makeMove(moveToPlay);
            ply++;
        }

        // --- END TIMER & CALCULATE SPEED ---
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;

        auto [over, result] = board.isGameOver();

        Color winner = Color::NONE; // stays none if draw

        if (result == GameResult::LOSE)
        {
            // sideToMove is checkmated, so opposite side won
            winner = (board.sideToMove() == Color::WHITE)
                ? Color::BLACK
                : Color::WHITE;
        }
        
        double movesPerSec = (elapsed.count() > 0) ? (ply / elapsed.count()) : 0.0;

        std::cout << "Game " << i << ": " << startingFen 
                  << " | plies: " << ply 
                  << " | speed: " << std::fixed << std::setprecision(2) << movesPerSec << " plies/s" 
                  << std::endl;

        writeGameToBinary("data/selfplay.bin", game, winner);
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
            std::cout << "option name Hash type spin default " << DEFAULT_HASH_MIB << " min 1 max 65536" << std::endl;
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
            //std::cout << "Evaluation for this position: " << evaluate_network(board) << std::endl;
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