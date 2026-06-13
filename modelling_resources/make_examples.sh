
awk -F': ' '
BEGIN {
    total = 0
    Z = 0
}

NF == 2 {
    X = NR
    N = $1 + 0
    phrase = $2

    n = split(phrase, parts, ",")

    for (y = 1; y <= N; y++) {
        Z++
        for (k = 1; k <= n; k++) {
            gsub(/^ +| +$/, "", parts[k])
            printf("%s(person_%d_%d).\n", parts[k], X, y)
        }
        # stroke line
        printf("stroke(person_%d_%d).\n", X, y)

        total++
    }
}

END {
    missing = 900 - total

    for (i = 1; i <= missing; i++) {
        printf("stroke(person_%d).\n", Z + i)
    }
}
' Claus_summary.txt
