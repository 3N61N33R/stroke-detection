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

    # Run ProbLog
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
            echo "query(stroke_or_tia)."
        } | problog --combine $model -
    )

    # Extract probability 
    prob=$(echo "$output" | awk '/^stroke_or_tia:/ {print $2}')

    # Final output
    echo "$prob :: stroke_or_tia :- $label."
done
