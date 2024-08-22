## How to build the training data for CrossDocked dataset

Step 1: `cd` to the `data` folder in `TamGen` repo if you are not in this folder.

Step 2: run `bash build_crossdocked_dataset.sh`

After running `build_crossdocked_dataset.sh`  in the `data` folder, you should get 
   ```bash
   data
├── build_crossdocked_dataset.sh
└── crossdocked
    ├── bin
    │   ├── dict.m1.txt
    │   ├── dict.tg.txt
    │   ├── src
    │   ├── structure-files-test
    │   ├── structure-files-train
    │   ├── structure-files-valid
    │   ├── test-coordinates.orig.pkl
    │   ├── test-coordinates.pkl
    │   ├── test-info.csv
    │   ├── test.tg-m1.m1.bin
    │   ├── test.tg-m1.m1.idx
    │   ├── test.tg-m1.tg.bin
    │   ├── test.tg-m1.tg.idx
    │   ├── train-coordinates.orig.pkl
    │   ├── train-coordinates.pkl
    │   ├── train-info.csv
    │   ├── train.tg-m1.m1.bin
    │   ├── train.tg-m1.m1.idx
    │   ├── train.tg-m1.tg.bin
    │   ├── train.tg-m1.tg.idx
    │   ├── valid-coordinates.orig.pkl
    │   ├── valid-coordinates.pkl
    │   ├── valid-info.csv
    │   ├── valid.tg-m1.m1.bin
    │   ├── valid.tg-m1.m1.idx
    │   ├── valid.tg-m1.tg.bin
    │   └── valid.tg-m1.tg.idx
    └── raw
        ├── crossdocked_pocket10
        ├── crossdocked_pocket10.tar.gz
        └── split_by_name.pt
   ```
   We also update a copy of the processed data, which can be accessed via the link provided in the manuscript.