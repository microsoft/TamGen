# I set two pockets, one is 10A, the other is 12A
for thr in "10" "12"; do
python ../scripts/build_data/prepare_pdb_ids_center_scaffold.py \
7vh8_out.csv test \
-o 7vh8-bin/t$thr -t $thr \
--scaffold-file 'seed_cmpd_7vh8.txt' \
--pdb-path "./"
done