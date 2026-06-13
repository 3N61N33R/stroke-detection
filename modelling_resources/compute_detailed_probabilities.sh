#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 model_file"
    exit 1
fi

model="$1"

features=(weakness facial_palsy speech sensory visual)

for mask in {0..31}; do
    label=""

    # Build label
    for i in "${!features[@]}"; do
        feature=${features[$i]}
        if (( (mask >> i) & 1 )); then
            label+="${feature},"
        else
            label+="\\+${feature},"
        fi
    done
    label=${label%,}

    # Run ProbLog with THREE queries
    output=$(
        {
            for i in "${!features[@]}"; do
                feature=${features[$i]}
                if (( (mask >> i) & 1 )); then
                    echo "evidence(${feature})."
                else
                    echo "evidence(\\+ ${feature})."
                fi
            done

            echo "query(tia)."
            echo "query(minor)."
            echo "query(major)."

        } | problog --combine $model -
    )

    # Extract probabilities
    read tia minor major < <(
	echo "$output" | awk '
             /tia:/   {t=$NF}
             /minor:/ {m=$NF}
             /major:/ {M=$NF}
             END {print t, m, M}
    '
)
    # Final output
    echo "$tia :: tia ; $minor :: minor ; $major :: major :- $label."
done
