import chess
import chess.engine
import math
import os
import sys
import concurrent.futures
import threading
import random
import time  # Added for tracking ETA

# ==================== CONFIGURATION ====================
TARGET_ENGINE_PATH = r"../neural-net4"
STOCKFISH_PATH = r"/usr/bin/stockfish"

GAMES_PER_LEVEL = 16         # Number of games per Elo step (keep even for color balance)
TIME_LIMIT_PER_MOVE = 1    # Seconds allowed per move
NODE_LIMIT_PER_MOVE = 99999
CONCURRENT_GAMES = 5         # Number of games to run simultaneously
# =======================================================

# Thread lock to prevent overlapping output lines in the terminal
print_lock = threading.Lock()

def calculate_overall_elo(game_history):
    """
    Finds the target Elo by solving the Bradley-Terry/Elo equation:
    Sum of expected scores against all opponents == Actual total score.
    """
    if not game_history:
        return 0
        
    total_actual_score = sum(game["score"] for game in game_history)
    
    low_elo = 0.0
    high_elo = 4000.0
    
    for _ in range(100):  
        mid_elo = (low_elo + high_elo) / 2
        expected_score = 0.0
        
        for game in game_history:
            expected_score += 1.0 / (1.0 + 10.0 ** ((game["opponent_elo"] - mid_elo) / 400.0))
            
        if expected_score < total_actual_score:
            low_elo = mid_elo
        else:
            high_elo = mid_elo

        random.shuffle(game_history)
            
    return round(mid_elo)

def get_progress_bar(current, total, bar_length=20):
    """Generates a text-based visual progress bar for total games."""
    fraction = current / total if total > 0 else 0
    filled_length = int(round(bar_length * fraction))
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    return f"[{bar}] {current}/{total} ({fraction * 100:.1f}%)"

def play_single_game(elo, target_is_white):
    """
    Runs a single game instance. Spawns dedicated engine processes locally
    to ensure absolute thread isolation.
    """
    target_engine = None
    sf_engine = None
    try:
        target_engine = chess.engine.SimpleEngine.popen_uci(TARGET_ENGINE_PATH)
        sf_engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        # CRITICAL: Limit internal engine threads to 1 to prevent CPU thrashing
        try:
            sf_engine.configure({"Threads": 1})
            target_engine.configure({"Threads": 1})
        except Exception:
            pass # Fallback if target engine doesn't support Threads parameter
            
        sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        
        board = chess.Board()

        for i in range(8):
            board.push(random.choice(list(board.legal_moves)))

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                current_engine = target_engine if target_is_white else sf_engine
            else:
                current_engine = sf_engine if target_is_white else target_engine
                
            #result = current_engine.play(board, chess.engine.Limit(time=TIME_LIMIT_PER_MOVE, nodes=NODE_LIMIT_PER_MOVE))
            result = current_engine.play(board, chess.engine.Limit(time=TIME_LIMIT_PER_MOVE))
            board.push(result.move)
            
        res_str = board.result()
        game_score = 0.0
        result_type = "D" # Win, Loss, Draw
        
        if res_str == "1-0":
            if target_is_white:
                game_score = 1.0
                result_type = "W"
            else:
                result_type = "L"
        elif res_str == "0-1":
            if not target_is_white:
                game_score = 1.0
                result_type = "W"
            else:
                result_type = "L"
        else:
            game_score = 0.5
            result_type = "D"
            
        return {"elo": elo, "score": game_score, "result_type": result_type, "error": None}
        
    except Exception as e:
        return {"elo": elo, "score": 0.0, "result_type": "E", "error": str(e)}
    finally:
        if target_engine:
            target_engine.quit()
        if sf_engine:
            sf_engine.quit()

def run_ladder_match():
    if not os.path.exists(TARGET_ENGINE_PATH) or not os.path.exists(STOCKFISH_PATH):
        print("Error: Please verify your engine executable paths.")
        return

    elo_levels = list(range(1320, 2320, 200)) 
    game_history = []
    
    # Tracking map for individual level distributions
    level_stats = {elo: {"wins": 0, "losses": 0, "draws": 0, "score": 0.0} for elo in elo_levels}

    total_planned_games = len(elo_levels) * GAMES_PER_LEVEL
    games_played = 0

    print("Starting concurrent multi-level Elo estimation ladder...")
    print(f"Testing tiers: {elo_levels}")
    print(f"Total planned games: {total_planned_games}")
    print(f"Running {CONCURRENT_GAMES} games simultaneously...\n" + "="*50)

    # Flatten and build all structured game tasks
    tasks = []
    for elo in elo_levels:
        for game_num in range(GAMES_PER_LEVEL):
            target_is_white = (game_num % 2 == 0)
            tasks.append((elo, target_is_white))

    # Record benchmark start time
    start_time = time.time()

    # Manage execution pools asynchronously
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_GAMES) as executor:
        # Submit all game tasks to the worker queue
        future_to_game = {
            executor.submit(play_single_game, elo, target_white): (elo, target_white) 
            for elo, target_white in tasks
        }
        
        # Process results immediately as they finish
        for future in concurrent.futures.as_completed(future_to_game):
            res = future.result()
            games_played += 1
            elo = res["elo"]
            
            if res["result_type"] == "E":
                with print_lock:
                    print(f"\n[ERROR] Game at {elo} Elo failed: {res['error']}")
                continue
                
            score = res["score"]
            rtype = res["result_type"]
            target_color = "White" if future_to_game[future][1] else "Black"
            
            # Store data safely in main thread execution
            game_history.append({"opponent_elo": elo, "score": score})
            level_stats[elo]["score"] += score
            if rtype == "W":
                level_stats[elo]["wins"] += 1
            elif rtype == "L":
                level_stats[elo]["losses"] += 1
            elif rtype == "D":
                level_stats[elo]["draws"] += 1
                
            # Dynamic ETA Calculation
            elapsed_time = time.time() - start_time
            games_remaining = total_planned_games - games_played
            avg_time_per_game = elapsed_time / games_played
            eta_seconds = avg_time_per_game * games_remaining
            
            # Format ETA into readable H:M:S format
            if games_remaining == 0:
                eta_str = "0s"
            else:
                mins, secs = divmod(int(eta_seconds), 60)
                hrs, mins = divmod(mins, 60)
                if hrs > 0:
                    eta_str = f"{hrs}h {mins}m {secs}s"
                elif mins > 0:
                    eta_str = f"{mins}m {secs}s"
                else:
                    eta_str = f"{secs}s"
                
            # Log progress cleanly without messing up multiple threads
            with print_lock:
                print(f"Finished: SF {elo:<4} (Target={target_color:<5}) -> Result: {rtype} | {get_progress_bar(games_played, total_planned_games)} | ETA: {eta_str}")

    # --- Final Assessment Generation ---
    final_estimated_elo = calculate_overall_elo(game_history)
    total_games = len(game_history)
    total_score = sum(g["score"] for g in game_history)
    
    print("\n" + "="*60)
    print("summary")
    print("="*60)
            
    print("-"*60)
    if total_games > 0:
        global_pct = (total_score / total_games) * 100
        print(f"performance: {total_score}/{total_games} games ({global_pct:.1f}%)")
        print(f"estimated engine elo: {final_estimated_elo}")
    print("-"*60)
    print(f"{'sf elo':<18}{'record (w-l-d)':<18}{'score percentage'}")
    for elo in elo_levels:
        stats = level_stats[elo]
        record_str = f"{stats['wins']}-{stats['losses']}-{stats['draws']}"
        pct = (stats["score"] / GAMES_PER_LEVEL) * 100
        print(f"{elo:<18}{record_str:<18}{pct:.1f}%")
    print("="*60)

if __name__ == "__main__":
    run_ladder_match()