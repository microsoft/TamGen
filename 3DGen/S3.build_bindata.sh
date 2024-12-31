# I set two pockets, one is 10A, the other is 12A
pdbid=$1
pdbupper=$(echo ${pdbid} | tr '[:lower:]' '[:upper:]')

for thr in "10"; do
python ../scripts/build_data/prepare_pdb_ids_center_scaffold.py \
./pdb_storage/index_simple_${pdbupper}.csv test \
-o bin/${pdbid}/t$thr -t $thr \
--scaffold-file seed_cmpd_${pdbid}.txt \
--pdb-path "./pdb_storage"
done