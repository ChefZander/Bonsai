echo "[1/4] Cleaning profiles folder..."

mkdir -p build
rm -rf build/profiles
mkdir -p build/profiles

echo "[2/4] Building PGO Build..."
g++ -O3 -march=native -std=c++20 -flto -ffast-math -funroll-loops -fprofile-generate -fprofile-dir=build/profiles bonsai.cpp -o build/bonsai-pgo

echo "[3/4] Profiling... (~20s)"
printf "%s\n" \
  "position startpos" \
  "go movetime 4000" \
  "position fen r1bqk2r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 4 4" \
  "go movetime 4000" \
  "position fen r2q1rk1/pb1nbppp/1p2p3/2pp4/3P4/2PBPN2/PP1N1PPP/R2QR1K1 w - - 4 11" \
  "go movetime 4000" \
  "position fen 3r1rk1/pp3ppp/2n5/2b5/4b3/5N2/PPP2PPP/R1B2RK1 w - - 0 12" \
  "go movetime 4000" \
  "position fen 8/2p5/3p4/kp1P4/p1P5/1P6/P7/1K6 w - - 0 1" \
  "go movetime 4000" \
  "quit" | ./build/bonsai-pgo

echo "[4/4] Rebuilding with Profile..."
g++ -O3 -march=native -std=c++20 -flto -ffast-math -funroll-loops -fprofile-use -fprofile-dir=build/profiles bonsai.cpp -o build/bonsai-pgo