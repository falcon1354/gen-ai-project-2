"""
sql_parse.py
------------
Convert model text like:
  select <c2> where <c0> = terrence ross

into WikiSQL dict:
  {"sel": 2, "agg": 0, "conds": [[0, 0, "terrence ross"]]}

and into readable SQL using real column names.
"""

import re


AGG_NAMES = ["", "max", "min", "count", "sum", "avg"]
COND_OPS = ["=", ">", "<", "op"]


def parse_col_token(token):
    """
    '<c12>' -> 12
    return None if not a column token
    """
    token = token.strip()
    if token.startswith("<c") and token.endswith(">"):
        inside = token[2:-1]
        if inside.isdigit():
            return int(inside)
    return None


def parse_generated_sql(text):
    """
    Parse generated text into a WikiSQL-style dict.
    Returns (sql_dict, error_message).
    On success error_message is None.
    """
    if text is None:
        return None, "empty text"

    # clean
    text = text.strip().lower()
    text = text.replace("<s>", " ").replace("</s>", " ").replace("<pad>", " ")
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return None, "empty after clean"

    tokens = text.split(" ")

    if len(tokens) == 0 or tokens[0] != "select":
        return None, "does not start with select"

    i = 1  # position after 'select'

    # optional aggregation
    agg = 0
    if i < len(tokens) and tokens[i] in AGG_NAMES[1:]:
        # find agg index
        a = 1
        while a < len(AGG_NAMES):
            if AGG_NAMES[a] == tokens[i]:
                agg = a
                break
            a = a + 1
        i = i + 1

    # selected column
    if i >= len(tokens):
        return None, "missing select column"

    sel = parse_col_token(tokens[i])
    if sel is None:
        return None, "select column is not <cN>"
    i = i + 1

    conds = []

    # optional where
    if i < len(tokens):
        if tokens[i] != "where":
            return None, "expected where or end, got: " + tokens[i]
        i = i + 1

        # parse conditions: <cN> op value value ... [and ...]
        while i < len(tokens):
            if i >= len(tokens):
                break

            col = parse_col_token(tokens[i])
            if col is None:
                return None, "condition column is not <cN>"
            i = i + 1

            if i >= len(tokens):
                return None, "missing operator"

            op_token = tokens[i]
            if op_token not in COND_OPS:
                return None, "bad operator: " + op_token
            op = COND_OPS.index(op_token)
            i = i + 1

            # value = everything until 'and' or end
            if i >= len(tokens):
                return None, "missing value"

            value_parts = []
            while i < len(tokens) and tokens[i] != "and":
                value_parts.append(tokens[i])
                i = i + 1

            if len(value_parts) == 0:
                return None, "empty value"

            value = " ".join(value_parts)
            conds.append([col, op, value])

            # skip 'and'
            if i < len(tokens) and tokens[i] == "and":
                i = i + 1

    sql = {
        "sel": sel,
        "agg": agg,
        "conds": conds,
    }
    return sql, None


def sql_to_readable(sql, headers, table_name="table"):
    """
    Turn WikiSQL dict + real headers into readable SQL string.
    """
    if sql is None:
        return None

    sel = sql["sel"]
    agg = sql["agg"]
    conds = sql["conds"]

    if sel < 0 or sel >= len(headers):
        return None

    col_name = headers[sel]

    if AGG_NAMES[agg] == "":
        select_part = col_name
    else:
        select_part = AGG_NAMES[agg].upper() + "(" + col_name + ")"

    query = "SELECT " + select_part + " FROM " + table_name

    if len(conds) > 0:
        pieces = []
        i = 0
        while i < len(conds):
            c_col, c_op, c_val = conds[i]
            if c_col < 0 or c_col >= len(headers):
                return None
            name = headers[c_col]
            op = COND_OPS[c_op]

            # quote strings; leave plain numbers unquoted
            val_str = str(c_val)
            if is_number(val_str):
                right = val_str
            else:
                right = "'" + val_str.replace("'", "''") + "'"

            pieces.append(name + " " + op + " " + right)
            i = i + 1
        query = query + " WHERE " + " AND ".join(pieces)

    return query


def is_number(text):
    try:
        float(text)
        return True
    except Exception:
        return False


def sql_logical_equal(a, b):
    """
    Compare two WikiSQL dicts for logical-form match.
    Condition order does not matter. Values compared lowercased as strings.
    """
    if a is None or b is None:
        return False

    if a["sel"] != b["sel"]:
        return False
    if a["agg"] != b["agg"]:
        return False
    if len(a["conds"]) != len(b["conds"]):
        return False

    set_a = set()
    for col, op, val in a["conds"]:
        set_a.add((col, op, str(val).lower()))

    set_b = set()
    for col, op, val in b["conds"]:
        set_b.add((col, op, str(val).lower()))

    return set_a == set_b


if __name__ == "__main__":
    text = "select <c2> where <c0> = terrence ross"
    sql, err = parse_generated_sql(text)
    print("sql", sql, "err", err)

    headers = ["player", "no.", "nationality", "position"]
    print(sql_to_readable(sql, headers))
