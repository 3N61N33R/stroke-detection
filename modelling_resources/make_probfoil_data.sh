#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 input_file"
    exit 1
fi

awk '
{
    # extract probability safely (from start of line)
    if (!match($0, /^[0-9.]+/)) {
        next   # skip malformed lines
    }
    prob = substr($0, RSTART, RLENGTH)

    # split on ":-"
    if (split($0, parts, ":-") < 2) {
        next   # skip malformed lines
    }

    cond = parts[2]

    # clean trailing dot and spaces
    sub(/[.[:space:]]+$/, "", cond)

    p = NR

    # print weighted stroke fact
    printf("%s :: stroke(person%d).\n", prob, p)

    # split literals
    n = split(cond, lits, ",")

    for (i = 1; i <= n; i++) {
        lit = lits[i]
        gsub(/^ +| +$/, "", lit)

        # skip negated literals
        if (lit ~ /^\\\+/) {
            continue
        }

        # skip empty tokens
        if (lit == "") {
            continue
        }

        printf("%s(person%d).\n", lit, p)
    }
}
' "$1"
