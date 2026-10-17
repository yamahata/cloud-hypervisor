#!/bin/sh
# Copyright (c) 2026 Crusoe Energy Systems LLC
#
# SPDX-License-Identifier: Apache-2.0
#
# Queries every --query-gpu property nvidia-smi lists, one per call,
# and prints tab-separated lines the test_infra crate parses:
#   CH_SWEEP_LISTED <count>      how many properties --help-query-gpu listed
#   CH_SWEEP <property> <GPU index> <value>
#   CH_SWEEP_FAILED <property> <first line of nvidia-smi's output>
properties=$(nvidia-smi --help-query-gpu |
    sed -n 's/^"\([A-Za-z0-9_.-]*\)".*/\1/p' |
    sort -u)
printf 'CH_SWEEP_LISTED\t%s\n' "$(printf '%s\n' "$properties" | grep -c .)"
[ -z "$properties" ] || printf '%s\n' "$properties" |
    while read -r property; do
        if out=$(nvidia-smi --query-gpu=index,"$property" \
            --format=csv,noheader,nounits 2>&1 </dev/null); then
            printf '%s\n' "$out" | while IFS=, read -r index value; do
                printf 'CH_SWEEP\t%s\t%s\t%s\n' "$property" "$index" "$value"
            done
        else
            printf 'CH_SWEEP_FAILED\t%s\t%s\n' "$property" \
                "$(printf '%s' "$out" | head -n 1)"
        fi
    done
