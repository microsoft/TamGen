import os.path
from contextlib import redirect_stdout
from io import StringIO

import mlflow.pyfunc
from fairseq import options
from fairseq.data.inmemorycache import load_data
from generate_multiseed import main
from postprocess import post_process

artifacts = {
    "checkpoint": "checkpoints/crossdock_pdb_A10/checkpoint_best.pt",
    "gpt_dir": "gpt_model",
}


class TamGenModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.ckpt = context.artifacts["checkpoint"]
        self.gpt_ckpt = os.path.join(context.artifacts["gpt_dir"], "checkpoint_best.pt")

    def predict(self, context, model_input, params=None):
        load_data(model_input)

        input_args = [
            "inmemorycache",
            "-s", "tg",
            "-t", "m1",
            "--task", "translation_coord",
            "--path", self.ckpt,
            "--model-overrides", f"{{'pretrained_gpt_checkpoint': '{self.gpt_ckpt}'}}",
            "--gen-subset", "test",
            "--beam", "20",
            "--nbest", "20",
            "--max-tokens", "1024",
            "--seed", "1",
            "--sample-beta", "1.0",
            "--use-src-coord",
            "--max-seed", "3",
            "--gen-vae",
            "--dataset-impl", "inmemorycache",
        ]

        parser = options.get_generation_parser()
        args = options.parse_args_and_arch(parser, input_args)

        output = main(args)

        result = { "result": post_process(output) }

        return result


mlflow_pyfunc_model_path = "/home/biran/foundry/mlflow/tamgen_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    python_model=TamGenModelWrapper(),
    artifacts=artifacts,
    conda_env={
        "name": "tamgen-mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.9",
            "pip<=24.2",
            "git",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle==3.1.0",
                    "biopython==1.84",
                    "dm-tree==0.1.8",
                    "editdistance==0.8.1",
                    "fairseq==0.8.0",
                    "fy-common-ext==0.1.0",
                    "numpy==1.26.4",
                    "rdkit==2024.3.1",
                    "scipy==1.13.1",
                    '--find-links https://download.pytorch.org/whl/torch_stable.html',
                    'torch==2.3.0+cu121',
                    '--find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html',
                    'torch-cluster==1.6.3+pt23cu121',
                    '-e git+https://github.com/microsoft/TamGen.git@biran/deploy#egg=fairseq',
                ],
            },
        ],
    },
)
