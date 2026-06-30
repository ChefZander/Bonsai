cutechess-cli \
  -engine name="64hl1" cmd="./build/bonsai-64hl1" \
  -engine name="24hl1" cmd="./build/bonsai-24hl1" \
  -each proto=uci st=0.1 timemargin=9999 \
  -openings file="./data/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -repeat \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 4 \
  -rounds 10000 \
  -ratinginterval 5