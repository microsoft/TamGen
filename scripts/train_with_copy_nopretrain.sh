SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

FAIRSEQ="$(dirname "$SCRIPT_DIR")"
cd $FAIRSEQ
# pip install --editable .[chem]


Layers=4
EmbedDims=256
LR=3e-5
DP=0.2
RunTrain=true
RunGenerate=true
datadir=dataset
trainset=train
validset=valid
testset=test-crossdocked
ExtraArgs=()
beam=20
seed=1
beta=1
DBS=false
groups=$beam
strength=0.5
savedir=checkpoints/

while [[ $# -gt 0 ]]; do
    case $1 in
        -L | --Layers )
            Layers=$2
            shift 2
            ;;
        -Dim )
            EmbedDims=$2
            shift 2
            ;;
        -LR | --LearningRate )
            LR=$2
            shift 2
            ;;
        -DP | --DropOut )
            DP=$2
            shift 2
            ;;
        --no-train )
            RunTrain=false
            shift
            ;;
        --no-generate )
            RunGenerate=false
            shift
            ;;
        --beam )
            beam=$2
            shift 2
            ;;
        --seed )
            seed=$2
            shift 2
            ;;
        --beta )
            beta=$2
            shift 2
            ;;
        --DBS )
            DBS=true
            shift
            ;;
        --groups )
            groups=$2
            shift 2
            ;;
        --strength )
            strength=$2
            shift 2
            ;;
        -D | --datadir )
            datadir=$2
            shift 2
            ;;
        --trainset )
            trainset=$2
            shift 2
            ;;
        --validset )
            validset=$2
            shift 2
            ;;
        --testset )
            testset=$2
            shift 2
            ;;
        --savedir )
            savedir=$2
            shift 2
            ;;
        *)
            ExtraArgs+=("$1")
            shift
	        ;;
    esac
done



SUFFIX=$(echo ${ExtraArgs[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g')
if [ -n "$SUFFIX" ]; then
    savedir=${savedir}-${SUFFIX}
fi
mkdir -p $savedir


if $RunTrain; then
    set -x
    python $FAIRSEQ/train.py \
    ${datadir} \
    -s tg -t m1 \
    --task translation_coord \
    --ddp-backend no_c10d \
    --arch transformer_3d \
    --share-decoder-input-output-embed \
    --criterion label_smoothed_cross_entropy_with_vae \
    --batch-size 256 \
    --dropout $DP \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler "inverse_sqrt" \
    --weight-decay 1e-4 \
    --lr $LR \
    --warmup-updates 1000 \
    --max-tokens 4096 \
    --encoder-layers $Layers \
    --encoder-embed-dim $EmbedDims \
    --encoder-ffn-embed-dim $((4*EmbedDims)) \
    --encoder-attention-heads 8 \
    --decoder-layers $Layers \
    --decoder-embed-dim $EmbedDims \
    --decoder-ffn-embed-dim $((4*EmbedDims)) \
    --decoder-attention-heads 8 \
    --no-token-positional-embeddings \
    --dist-attn \
    --move-to-origin \
    --random-rotation \
    --max-epoch 200 \
    --update-freq 1 \
    --save-dir $savedir \
    --train-subset $trainset \
    --valid-subset $validset \
    --vae \
    --use-src-coord \
    ${ExtraArgs[@]}| tee -a $savedir/log
    set +x
fi

for ii in {1..300}; do
    if ((ii % 50 != 0)); then
        rm -rf $savedir/checkpoint$ii.pt
    fi
done

cp -r $savedir /blob/yinxia/wu2/sandbox

if $RunGenerate; then
    GenExtraArgs=()
    function postprocess() {
                grep ^H | cut -f1,3- | tr -d ' '
            }

    task=translation_coord
    if $DBS ; then
        outputdir=$savedir/$dataset/${testset}/output_b${beam}_s${seed}_beta${beta}_DBS
        GenExtraArgs+=(--diverse-beam-groups $groups --diverse-beam-strength $strength)
    else
        outputdir=$savedir/$dataset/${testset}/output_b${beam}_s${seed}_beta${beta}
    fi
    mkdir -p $outputdir
    score=$outputdir/scores

    set -x
    python $FAIRSEQ/generate.py \
    ${datadir} \
    -s tg -t m1 \
    --task $task \
    --path $savedir/checkpoint_best.pt \
    --gen-subset $testset \
    --beam $beam --nbest $beam --max-tokens 1024 \
    --seed $seed --sample-beta $beta \
    --use-src-coord \
    ${GenExtraArgs[@]} | \
    postprocess > $outputdir/output.txt
    set +x

        

fi
