import sentencepiece
from typing import List, Union


class Tokenizer:
    def __init__(self):
        self.spm = sentencepiece.SentencePieceProcessor(
            model_file="checkpoint/sentencepiece.joint.bpe.model"
        )

    def _encode(self, sentence: str) -> str:
        return " ".join(self.spm.Encode(sentence, out_type=str))

    def _decode(self, model_output: List[str]) -> str:
        return "".join(model_output).replace("▁", "")

    def encode(self, input: Union[List[str], str]) -> Union[List[str], str]:
        if type(input) == List:
            return [self._encode(sentence) for sentence in input]
        else:
            return self._encode(input)


