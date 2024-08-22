"""Run the decomposed RoBERTa model."""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from fairseq.models.roberta import RobertaModel, DecomposedRobertaModel


PATH = r'\\msralpa\Users\v-yaf\public\DataTransfer\GitProjects\LightBERT\checkpoints\roberta.large'


def _test_roberta(model):
    model.eval()

    tokens = model.encode('Hello world!')
    assert tokens.tolist() == [0, 31414, 232, 328, 2]
    print(model.decode(tokens))     # 'Hello world!'

    model.register_classification_head('new_task', num_classes=3)
    logprobs = model.predict('new_task', tokens)
    print(logprobs)


def _load_normal_roberta():
    roberta = RobertaModel.from_pretrained(PATH, checkpoint_file='model.pt')
    _test_roberta(roberta)


def _load_d_roberta():
    d_roberta = DecomposedRobertaModel.from_pretrained(PATH, checkpoint_file='model_decomposed.pt')
    _test_roberta(d_roberta)


def main():
    _load_normal_roberta()
    _load_d_roberta()


if __name__ == "__main__":
    main()
