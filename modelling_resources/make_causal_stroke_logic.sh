
awk -F': ' '
BEGIN {
    i = 0
}

NF == 2 {
    count = $1 + 0
    phrase = $2

    if (!(phrase in id)) {
        i++
        id[phrase] = i
        phrases[i] = phrase
    }

    sum[phrase] += count
}

END {
    # Line 1
    line = ""
    for (j = 1; j <= i; j++) {
        p = phrases[j]
        if (j > 1) line = line "; "
        line = line sprintf("(%d/900)::c_%d", sum[p], j)
    }
    print line

    # Line 2 (no quotes)
    print ":- stroke."

    # Line 3 (empty)
    print ""

    # Following lines
    for (j = 1; j <= i; j++) {
        p = phrases[j]
        n = split(p, parts, ",")

        for (k = 1; k <= n; k++) {
            gsub(/^ +| +$/, "", parts[k])  # trim spaces
            printf("%s :- c_%d.\n", parts[k], j)
        }
    }
}
' Claus_detailed_summary.txt
