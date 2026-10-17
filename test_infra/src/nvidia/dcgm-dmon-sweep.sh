#!/bin/sh
# Copyright (c) 2026 Crusoe Energy Systems LLC
#
# SPDX-License-Identifier: Apache-2.0
#
# Samples every field "dcgmi dmon -l" lists once, one per call, and
# prints tab-separated lines the test_infra crate parses:
#   CH_DMON_LISTED <count>                fields listed outside the skipped ranges
#   CH_DMON <field ID> <name> <entity> <value>
#   CH_DMON_NOSAMPLE <field ID> <name>    dcgmi exited 0 with no sample row
#   CH_DMON_FAILED <field ID> <name> <first line of its output>
# Rate and bandwidth fields are skipped: they need several samples a second
# apart. The ID ranges are DCGM 4.4.2's, the same in 4.7.
skipped='200-201 440-449 475-480 494-496 780-781 825-843 861-862 879-897 1001-1083 1100-1104'

# "<field ID> <long name>" of each listed field outside the skipped ranges.
fields=$(dcgmi dmon -l </dev/null | tr -d '|' |
    awk -v skipped="$skipped" '
        BEGIN { n = split(skipped, range, " ") }
        NF >= 2 && $NF ~ /^[0-9]+$/ {
            for (i = 1; i <= n; i++) {
                split(range[i], bound, "-")
                if ($NF + 0 >= bound[1] + 0 && $NF + 0 <= bound[2] + 0) next
            }
            print $NF, $1
        }' |
    sort -n -u -k1,1)
printf 'CH_DMON_LISTED\t%s\n' "$(printf '%s\n' "$fields" | grep -c .)"
[ -z "$fields" ] || printf '%s\n' "$fields" |
    while read -r id name; do
        if out=$(dcgmi dmon -e "$id" -c 1 -d 1 2>&1 </dev/null) &&
            ! printf '%s\n' "$out" | grep -qE '^[[:space:]]*Error'; then
            # dmon prints a header, then one row per sample and entity,
            # "GPU 0     <value>": keep the last sample of each entity. A
            # field that prints no sample row at all is reported as such, so
            # that every listed field has a line.
            printf '%s\n' "$out" | awk -v id="$id" -v name="$name" '
                /^[A-Za-z_]+ +[0-9]+/ {
                    key = $1 " " $2
                    $1 = ""
                    $2 = ""
                    sub(/^ +/, "")
                    sub(/ +$/, "")
                    last[key] = $0
                    rows++
                }
                END {
                    if (!rows) printf "CH_DMON_NOSAMPLE\t%s\t%s\n", id, name
                    for (key in last)
                        printf "CH_DMON\t%s\t%s\t%s\t%s\n", id, name, key, last[key]
                }'
        else
            printf 'CH_DMON_FAILED\t%s\t%s\t%s\n' "$id" "$name" \
                "$(printf '%s' "$out" | head -n 1)"
        fi
    done
