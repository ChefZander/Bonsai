cutechess-cli \
  -engine name="Optimized SmallNet 5 Inference" cmd="./build/optimized" \
  -engine name="Neural SillyNet 2" cmd="./build/neural-sillynet2" \
  -each proto=uci st=0.1 timemargin=9999 \
  -openings file="./data/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -repeat \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 4 \
  -rounds 10000 \
  -ratinginterval 5