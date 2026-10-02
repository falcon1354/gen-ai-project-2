"""
download_wikisql.py
-------------------
Downloads WikiSQL and extracts the data/ folder into WikiSQL/data.

Run once from the project root:
  python download_wikisql.py
"""

import os
import tarfile
import urllib.request


URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    wiki_dir = os.path.join(project_root, "WikiSQL")
    archive_path = os.path.join(wiki_dir, "data.tar.bz2")
    data_dir = os.path.join(wiki_dir, "data")

    if not os.path.exists(wiki_dir):
        os.makedirs(wiki_dir)

    # already extracted?
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_jsonl):
        print("WikiSQL data already present at:", data_dir)
        return

    print("Downloading WikiSQL data...")
    print("URL:", URL)
    urllib.request.urlretrieve(URL, archive_path)
    print("Saved archive:", archive_path)

    print("Extracting...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(path=wiki_dir)

    # after extract we expect WikiSQL/data/...
    if os.path.exists(train_jsonl):
        print("OK. Data is ready at:", data_dir)
    else:
        print("Extract finished, but train.jsonl was not found.")
        print("Please check the contents of:", wiki_dir)

    # optional: remove archive to save space
    # os.remove(archive_path)


if __name__ == "__main__":
    main()
