cutechess-cli \
  -engine name="keep90" cmd="./build/bonsai-pgo-90" \
  -engine name="keep50" cmd="./build/bonsai-pgo-50" \
  -each proto=uci st=5 timemargin=9999 \
  -openings file="./data/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -repeat \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 4 \
  -rounds 10000 \
  -ratinginterval 5