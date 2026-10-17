#!/bin/sh
# Copyright (c) 2026 Crusoe Energy Systems LLC
#
# SPDX-License-Identifier: Apache-2.0
#
# The GPUs' error counters, one per line, compared before and after a test:
# the AER counters of every NVIDIA PCI function, then the error, ECC, replay,
# retired-page and row-remap lines of "nvidia-smi -q", each prefixed with its
# GPU and section (the sections repeat the labels), as in
#   00000008:00:00.0 / ECC Errors / Volatile / SRAM Correctable : 0
for device in /sys/bus/pci/devices/*; do
    [ "$(cat "$device/vendor" 2>/dev/null)" = 0x10de ] || continue
    for counter in aer_dev_correctable aer_dev_nonfatal aer_dev_fatal; do
        [ -r "$device/$counter" ] &&
            sed "s|^|${device##*/} $counter: |" "$device/$counter"
    done
done
# A failed nvidia-smi fails the script, which a pipe would hide.
q=$(nvidia-smi -q) || exit 1
printf '%s\n' "$q" | awk '
    /^GPU / {
        gpu = $2
        next
    }
    {
        line = $0
        sub(/^ +/, "", line)
        if (line == "") next
        depth = (length($0) - length(line)) / 4
        if (index(line, " : ") == 0) {
            heading[depth] = line
            next
        }
        if (tolower(line) !~ /replay|error|correctable|uncorr|remap|retired|pending|xid|ecc|fatal/) next
        prefix = gpu
        for (level = 1; level < depth; level++) prefix = prefix " / " heading[level]
        gsub(/ +: /, " : ", line)
        print prefix " / " line
    }'
