export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 

for L in "4" "6"; do
for lr in "1e-4" "5e-5" "2e-5"; do
for dr in "0.1" "0.2"; do
for beta in "1" "0.1"; do

savefolder="ghddi_frag_nopretrain_L${L}_lr${lr}_beta${beta}_dr${dr}"
bash scripts/train_with_copy_nopretrain.sh --savedir $savefolder -L $L --beta $beta --fp16 --datadir dataset/bin/ -LR $lr -DP $dr --no-generate
done
done
done
done
# bash scripts/train_with_copy_nopretrain.sh --savedir 001133 -L 5 --beta 0.1 --fp16 --datadir dataset/bin/
