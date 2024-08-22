SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
FAIRSEQ="$(dirname "$SCRIPT_DIR")"
datadir=dataset
dataset=test-crossdocked
ckpt=model/checkpoint_best.pt
savedir=output
ExtraArgs=()

cd $FAIRSEQ

beam=20
seed=0
beta=1

# Diverse Beam Search
DBS=false
groups=$beam
strength=0.5

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --dataset )
            dataset=$2
            shift 2
            ;;
        --model-path | --ckpt )
            ckpt=$2
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


stringContain() { [ -z "$1" ] || { [ -z "${2##*$1*}" ] && [ -n "$2" ];};}

function postprocess() {
                grep ^H | cut -f1,3- | tr -d ' '
            }

task=translation_coord

SUFFIX=$(echo ${ExtraArgs[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g')

    set -x
    if $DBS ; then
        outputdir=$savedir/${dataset}/output_b${beam}_s${seed}_beta${beta}_DBS
        ExtraArgs+=(--diverse-beam-groups $groups --diverse-beam-strength $strength)
    else
        outputdir=$savedir/${dataset}/output_b${beam}_s${seed}_beta${beta}
    fi
    if [ -n "$SUFFIX" ]; then
        outputdir=${outputdir}-${SUFFIX}
    fi
    mkdir -p $outputdir
    set +x
    score=$outputdir/scores

    set -x
    python $FAIRSEQ/generate.py \
    ${datadir} \
    -s tg -t m1 \
    --task $task \
    --path $ckpt \
    --gen-subset $dataset \
    --beam $beam --nbest $beam --max-tokens 1024 \
    --seed $seed --sample-beta $beta \
    --use-src-coord \
    ${ExtraArgs[@]} | \
    postprocess > $outputdir/output.txt
    set +x
