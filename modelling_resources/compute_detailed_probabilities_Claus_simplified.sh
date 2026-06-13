#!/usr/bin/env bash

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

        } | problog --combine detailed_causal_model_Claus_simplified.pl -
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
