"""
data_prep.py
------------
Turn WikiSQL files into simple (source, target) text pairs.

Source  = English question + table column names
Target  = SQL written as text, using <c0>, <c1>, ... for columns

Run this BEFORE tokenizer.py

Example source:
  what is the nationality of terrence ross ? <c0> player <c1> no . <c2> nationality <c3> position

Example target:
  select <c2> where <c0> = terrence ross
"""

import json
import os
import re


# WikiSQL aggregation names (index -> word)
# 0 means "no aggregation", so we write nothing extra
AGG_NAMES = ["", "max", "min", "count", "sum", "avg"]

# WikiSQL condition operators (index -> symbol)
COND_OPS = ["=", ">", "<", "op"]


def clean_text(text):
    """Make text lowercase and tidy up spaces."""
    text = str(text)
    text = text.lower()
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def load_tables(tables_path):
    """
    Read a *.tables.jsonl file into a dictionary.
    Key   = table id
    Value = list of column names (headers)
    """
    tables = {}

    with open(tables_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            table = json.loads(line)
            table_id = table["id"]
            headers = table["header"]

            # clean each column name
            clean_headers = []
            for h in headers:
                clean_headers.append(clean_text(h))

            tables[table_id] = clean_headers

    return tables


def make_source(question, headers):
    """
    Build the encoder input string.

    Format:
      <question> <c0> col0 <c1> col1 ...
    """
    question = clean_text(question)

    parts = [question]

    i = 0
    while i < len(headers):
        col_token = "<c" + str(i) + ">"
        parts.append(col_token)
        parts.append(headers[i])
        i = i + 1

    source = " ".join(parts)
    return source


def make_target(sql, headers):
    """
    Build the decoder target string from WikiSQL's sql object.

    Format examples:
      select <c2>
      select count <c1>
      select <c2> where <c0> = terrence ross
      select max <c3> where <c1> > 10 and <c0> = usa
    """
    sel = sql["sel"]
    agg = sql["agg"]
    conds = sql["conds"]

    # safety checks
    if sel < 0 or sel >= len(headers):
        return None

    if agg < 0 or agg >= len(AGG_NAMES):
        return None

    parts = ["select"]

    # add aggregation word if needed (skip empty string for agg=0)
    if AGG_NAMES[agg] != "":
        parts.append(AGG_NAMES[agg])

    # selected column as <cN>
    parts.append("<c" + str(sel) + ">")

    # WHERE conditions
    if len(conds) > 0:
        parts.append("where")

        cond_i = 0
        while cond_i < len(conds):
            cond = conds[cond_i]
            col_index = cond[0]
            op_index = cond[1]
            value = cond[2]

            if col_index < 0 or col_index >= len(headers):
                return None

            if op_index < 0 or op_index >= len(COND_OPS):
                return None

            # "and" between conditions
            if cond_i > 0:
                parts.append("and")

            parts.append("<c" + str(col_index) + ">")
            parts.append(COND_OPS[op_index])
            parts.append(clean_text(value))

            cond_i = cond_i + 1

    target = " ".join(parts)
    return target


def convert_split(examples_path, tables_path, out_path):
    """
    Convert one WikiSQL split (train/dev/test) into a pairs jsonl file.
    Each output line looks like:
      {"src": "...", "tgt": "...", "table_id": "..."}
    """
    tables = load_tables(tables_path)

    kept = 0
    skipped = 0

    out_file = open(out_path, "w", encoding="utf-8")

    with open(examples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue

            example = json.loads(line)
            table_id = example["table_id"]
            question = example["question"]
            sql = example["sql"]

            # skip if we cannot find the table
            if table_id not in tables:
                skipped = skipped + 1
                continue

            headers = tables[table_id]

            # skip empty tables
            if len(headers) == 0:
                skipped = skipped + 1
                continue

            source = make_source(question, headers)
            target = make_target(sql, headers)

            # skip bad / incomplete examples
            if source is None or target is None:
                skipped = skipped + 1
                continue

            if source == "" or target == "":
                skipped = skipped + 1
                continue

            row = {
                "src": source,
                "tgt": target,
                "table_id": table_id,
            }
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept = kept + 1

    out_file.close()

    print(os.path.basename(out_path) + ": kept " + str(kept) + ", skipped " + str(skipped))
    return kept, skipped


def main():
    # paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "WikiSQL", "data")
    out_dir = os.path.join(project_root, "starter", "processed")

    if not os.path.exists(data_dir):
        print("ERROR: WikiSQL data folder not found.")
        print("Expected folder:", data_dir)
        print("Download WikiSQL, extract it, and put the 'data' folder inside WikiSQL/")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # train
    convert_split(
        os.path.join(data_dir, "train.jsonl"),
        os.path.join(data_dir, "train.tables.jsonl"),
        os.path.join(out_dir, "train_pairs.jsonl"),
    )

    # dev
    convert_split(
        os.path.join(data_dir, "dev.jsonl"),
        os.path.join(data_dir, "dev.tables.jsonl"),
        os.path.join(out_dir, "dev_pairs.jsonl"),
    )

    # test (useful later for evaluation)
    convert_split(
        os.path.join(data_dir, "test.jsonl"),
        os.path.join(data_dir, "test.tables.jsonl"),
        os.path.join(out_dir, "test_pairs.jsonl"),
    )

    print("Done. Pairs saved in:", out_dir)


if __name__ == "__main__":
    main()
