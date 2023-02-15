import mediafire_dl
import requests
import tarfile
import os
import wget

#might need to get new links @ https://github.com/deep-spin/hallucinations-in-nmt

def main():
    print("Download script for OT for Hallucination detection")
    if not os.path.isfile("checkpoint/checkpoint_best.pt"):
        print("Downloading model... (800+ Mo)")

        mediafire_dl.download(
            "https://download2356.mediafire.com/l7lm29hmfdmg/mp5oim9hqgcy8fb/checkpoint_best.tar.xz",
            "checkpoint_best.tar.xz",
            quiet=False,
        )
        with tarfile.open("checkpoint_best.tar.xz") as f:
            f.extract("checkpoint_best.pt", "./checkpoint")

        os.remove("checkpoint_best.tar.xz")

    if not os.path.isdir("data/wmt18_de-en") or not os.listdir("data/wmt18_de-en"):
        print("Downloading data... (260+ Mo)")
        mediafire_dl.download(
            "https://download847.mediafire.com/s2x7sxq3kgcg/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz",
            "wmt18_de-en.tar.xz",
            False,
        )
        with tarfile.open("wmt18_de-en.tar.xz") as arch:
            for file in arch:
                arch.extract(file, "./data")

        os.remove("wmt18_de-en.tar.xz")

    if not os.path.isfile("data/annotated_corpus.csv"):
        print("Downloading annotated corpus ...")
        req = requests.get(
            "https://raw.githubusercontent.com/deep-spin/hallucinations-in-nmt/main/data/annotated_corpus.csv"
        )
        content = req.content
        with open("./data/annotated_corpus.csv", "wb") as csv_file:
            csv_file.write(content)

    if not os.path.isfile("checkpoint/sentencepiece.joint.bpe.model"):
        print("Downloading tokenizer...")
        spm_url = "https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.model"
        wget.download(spm_url, "./checkpoint/sentencepiece.joint.bpe.model")

    print('Everything is present')


if __name__ == "__main__":
    main()
