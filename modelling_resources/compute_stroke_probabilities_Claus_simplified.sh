
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
            echo "query(stroke)."
        } | problog --combine causal_model_Claus_simplified.pl -
    )

    # Extract probability 
    prob=$(echo "$output" | awk '/^stroke:/ {print $2}')

    # Final output
    echo "$prob :: stroke :- $label."
done
