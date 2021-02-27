import os
import json
import argparse
from tqdm.auto import tqdm
from vncorenlp import VnCoreNLP
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, List

CAPTIONS = {
    "coco_precomp": ["train_caps_vi.txt", "dev_caps_vi.txt"],
    "f30k_precomp": ["train_caps_vi.txt", "dev_caps_vi.txt"],
}


def load_rdrsegmenter():
    VNCORE_PATH = Path.home() / "vncorenlp"

    if not (VNCORE_PATH / "models/wordsegmenter/wordsegmenter.rdr").exists():
        Path(VNCORE_PATH / "models/wordsegmenter").mkdir(parents=True, exist_ok=True)
        os.system(
            "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -P %s"
            % str(VNCORE_PATH)
        )
        os.system(
            "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab -P %s"
            % str(VNCORE_PATH / "models/wordsegmenter")
        )
        os.system(
            "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr -P %s"
            % str(VNCORE_PATH / "models/wordsegmenter")
        )
        print("Downloaded model to %s" % str(VNCORE_PATH))

    return VnCoreNLP(
        str(VNCORE_PATH / "VnCoreNLP-1.1.1.jar"),
        annotators="wseg",
        max_heap_size="-Xmx500m",
    )


class ViVocabulary(object):
    def __init__(
        self,
        word2id: Optional[dict] = {},
        id2word: Optional[List[int]] = [],
        n_word: Optional[int] = 0,
    ):
        self.word2id = word2id
        self.id2word = id2word
        self.n_word = n_word
        self.tokenizer = self.load_tokenizer()

    def __len__(self) -> int:
        return self.n_word

    def __call__(self, word: str) -> int:
        if word not in self.word2id:
            return self.word2id['<unk>']
        return self.word2id[word]

    def add_word(self, word: str) -> int:
        """Add a word to Vocab list and return its index."""

        if word not in self.word2id:
            self.word2id[word] = self.n_word
            self.id2word.append(word)
            self.n_word += 1
        return self.word2id[word]

    def sent2id(self, sent: str) -> List[int]:
        tokens = self.tokenizer(sent.lower())[0]
        ids = [self('<start>')]
        ids += [self(token) for token in tokens]
        ids += [self('<end>')]
        return ids

    def load_tokenizer(self) -> Callable:
        VNCORE_PATH = Path.home() / "vncorenlp"

        if not (VNCORE_PATH / "models/wordsegmenter/wordsegmenter.rdr").exists():
            Path(VNCORE_PATH / "models/wordsegmenter").mkdir(parents=True, exist_ok=True)
            os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -P %s" % str(VNCORE_PATH))
            os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab -P %s" % str(VNCORE_PATH/'models/wordsegmenter'))
            os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr -P %s" % str(VNCORE_PATH/'models/wordsegmenter'))
            print("Downloaded model to %s" % str(VNCORE_PATH))
        
        rdr = VnCoreNLP(str(VNCORE_PATH/'VnCoreNLP-1.1.1.jar'), annotators="wseg", max_heap_size='-Xmx500m')

        return rdr.tokenize

    
    @classmethod
    def load(cls, file_path: str) -> "ViVocabulary":
        with open(file_path) as f:
            data = json.load(f)
        
        n_word = len(data['id2word'])
        print("Loaded %d vocabs from %s" % (n_word, file_path))
        return cls(data['word2id'], data['id2word'], n_word)
        
    
    def save(self, file_path, overwrite=True):
        data = {"word2id": self.word2id, "id2word": self.id2word}
        with open(file_path, "w") as f:
            json.dump(data, f)
        print("Saved %d vocabs to %s" % (self.n_word, file_path))

    @classmethod
    def build_from_txt(cls,
                       caption_path: str,
                       dataset: str,
                       tokenizer: Callable,
                       occurrences_thres: int = 4) -> "ViVocabulary":
        '''Read caption file and build a vocab list from those captions.'''
        
        counter = Counter()

        for file in CAPTIONS[dataset]:
            try:
                cap = Path(caption_path) / dataset / file
                with open(cap) as f:
                    data = f.readlines()
            except FileNotFoundError:
                print("%s does not exist!" % str(cap))

            print("Processing %s with %d captions" % (str(cap), len(data)))

            for cap in tqdm(data):
                tokenized_cap = tokenizer(cap.lower())
                counter.update(tokenized_cap[0])

        vocab = cls()
        vocab.add_word('<pad>')
        vocab.add_word('<start>')
        vocab.add_word('<end>')
        vocab.add_word('<unk>')

        
        for word, count in counter.items():
            if count > occurrences_thres:
                vocab.add_word(word)

        return vocab


def main(caption_path: str, dataset: str):
    rdrsegmenter = load_rdrsegmenter()
    vocab = ViVocabulary.build_from_txt(caption_path, dataset, rdrsegmenter.tokenize)
    vocab.save(Path(caption_path) / dataset / ("%s_vocab.json" % dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path", default="data")
    parser.add_argument("--dataset", default="f30k_precomp")
    opt = parser.parse_args()
    main(opt.caption_path, opt.dataset)
