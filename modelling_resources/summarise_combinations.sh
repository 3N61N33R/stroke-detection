
awk -F': ' '
NF==2 {
    sums[$2] += $1
}
END {
    for (p in sums)
        print sums[p] ": " p
}' Claus_combinations.txt
