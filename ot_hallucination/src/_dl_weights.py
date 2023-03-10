import mediafire_dl
import requests
import tarfile
import os
import wget

#If there is an error such that :FileNotFoundError: [Errno 2] No such file or directory: 'wmt18_de-en.tar.xz'
# It is because mediafire update its links. To do so, click on the links below, generate a new link, and replace it.

def main():
    print("Download script for OT for Hallucination detection")
    if not os.path.isfile("checkpoint/checkpoint_best.pt"):
        print("Downloading model... (800+ Mo)")

        mediafire_dl.download(
            "https://download1499.mediafire.com/2uid5yg67t7g/mp5oim9hqgcy8fb/checkpoint_best.tar.xz",
            "checkpoint_best.tar.xz",
            quiet=False,
        )
        with tarfile.open("checkpoint_best.tar.xz") as f:
            f.extract("checkpoint_best.pt", "./checkpoint")

        os.remove("checkpoint_best.tar.xz")

    if not os.path.isdir("data/wmt18_de-en") or not os.listdir("data/wmt18_de-en"):
        print("Downloading data... (260+ Mo)")
        mediafire_dl.download(
            "https://download847.mediafire.com/q917g6c8cz3gSs1zor4U1F_58g8Vobipq53Xx3zFIT5vQv7LlfPizFV8Krg73LD9pF9UMqMlChQ4ZpEplODfxKUP8A/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz",
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

    print('\n Everything is present')


if __name__ == "__main__":
    main()
