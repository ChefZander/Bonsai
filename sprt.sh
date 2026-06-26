cutechess-cli \
  -engine name="Neural SmallNet 5" cmd="./neural-smallnet5" \
  -engine name="Neural SillyNet 2" cmd="./neural-sillynet2" \
  -each proto=uci st=0.1 timemargin=9999 \
  -openings file="./sprt/UHO_Lichess_4852_v1.epd" format=epd order=random \
  -repeat \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 \
  -concurrency 4 \
  -rounds 10000 \
  -ratinginterval 5