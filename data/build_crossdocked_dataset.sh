CROSSDOCK_PATH=crossdocked/raw # You can also specify by yourself
OUTPUT_PATH=crossdocked/bin    # You can also specify by yourself
mkdir -p $CROSSDOCK_PATH
mkdir -p $OUTPUT_PATH
TAMGEN_FOLDER=$(dirname $(pwd))
cd $CROSSDOCK_PATH
pip install gdown
gdown https://drive.google.com/uc?id=1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY
gdown https://drive.google.com/uc?id=10KGuj15mxOJ2FBsduun2Lggzx0yPreEU
tar -xzvf crossdocked_pocket10.tar.gz
cd $TAMGEN_FOLDER
python scripts/build_data/prepare_crossdocked.py "data/"${CROSSDOCK_PATH} -o "data/"${OUTPUT_PATH}
cd data
python dump_coord.py $OUTPUT_PATH