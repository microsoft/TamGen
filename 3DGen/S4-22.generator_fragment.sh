export CUDA_VISIBLE_DEVICES=0
pdbid=$1
beam=20
testset="test"
task=translation_coord
ckpt="./checkpoints/crossdock_pdb_A10/checkpoint_best.pt"

results_folder="./3DGen/${pdbid}-fragment-results"

mkdir -p $results_folder

global_idx=0

for thr in "10"; do # you can use larger or smaller thresholds;
for beta in "1.0"; do # you can also try different betas like 0.1, etc

while IFS= read -r p
do

datadir="3DGen/bin/${pdbid}/t${thr}"
global_idx=$((global_idx+1))
log_file="logs_thr${thr}_beta${beta}_Globalid${global_idx}"

# this is the conditioned generation
python generate_multiseed.py \
${datadir} \
-s tg -t m1 \
--task $task \
--path $ckpt \
--gen-subset $testset \
--beam $beam --nbest $beam --max-tokens 1024 \
--seed 1 --sample-beta $beta \
--use-src-coord \
--max-seed 3 \
--gen-vae --prefix-string $p | tee -a ${results_folder}/vae_${log_file}_beta$beta

# this is the unconditioned generation
python generate_multiseed.py \
${datadir} \
-s tg -t m1 \
--task $task \
--path $ckpt \
--gen-subset $testset \
--beam $beam --nbest $beam --max-tokens 1024 \
--seed 1 --sample-beta $beta \
--max-seed 3 \
--use-src-coord --prefix-string $p | tee -a ${results_folder}/nonvae_${log_file}_beta$beta

done
done < "3DGen/4xli_prefix.smi.rev"
done