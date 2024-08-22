export CUDA_VISIBLE_DEVICES="8,9,10,11,12,13,14,15" 

for dr in "0.1" "0.2"; do
for L in "4" "6"; do
for lr in "3e-5" "5e-5"; do
for beta in "1" "0.1"; do

savefolder="ghddi_frag_pretrain_L${L}_lr${lr}_beta${beta}_dr${dr}"
bash scripts/train_with_copy.sh --savedir $savefolder -L $L --beta $beta --fp16 --datadir dataset/bin/ -LR $lr -DP $dr --no-generate

done
done
done
done
