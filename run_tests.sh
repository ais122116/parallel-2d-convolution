#!/usr/bin/env bash
# run_tests.sh - correctness + timing harness
#
# Every implementation must have the CLI:  <prog> input.bin filters.bin output.bin
# and print a line "time_ms: <value>".  GPU binaries that don't exist yet are skipped.
#
# Usage: ./run_tests.sh            (quick sweep)
#        ./run_tests.sh results.csv
set -u
CSV=${1:-results.csv}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

IMPLS=(./conv_naive ./conv_opt)   # add more optimized versions here

# Image sizes "M N": square / rectangular, power-of-2, even non-pow2, odd
SIZES=("64 64" "128 96" "100 100" "127 255" "33 17" "256 256" "300 201")
WIDTHS=(1 3 5 7)
# "C K" pairs from {3,4,8,16,32,64,127}
CK=("3 8" "4 4" "16 32" "127 3" "8 127")

echo "M,N,C,K,W,impl,time_ms,status" > "$CSV"
./cpu_reference --selftest || { echo "CPU selftest failed, aborting"; exit 1; }

pass=0; fail=0; seed=1
for s in "${SIZES[@]}"; do
  read -r M N <<< "$s"
  for W in "${WIDTHS[@]}"; do
    for ck in "${CK[@]}"; do
      read -r C K <<< "$ck"
      seed=$((seed + 1))
      ./gen_data "$M" "$N" "$C" "$K" "$W" "$seed" "$TMP/in.bin" "$TMP/f.bin" || exit 1

      t=$(./cpu_reference "$TMP/in.bin" "$TMP/f.bin" "$TMP/ref.bin" | awk '/time_ms/{print $2}')
      echo "$M,$N,$C,$K,$W,cpu,$t,REF" >> "$CSV"
      line="M=$M N=$N C=$C K=$K W=$W | cpu ${t}ms"

      for impl in "${IMPLS[@]}"; do
        [ -x "$impl" ] || continue
        name=$(basename "$impl")
        gt=$("$impl" "$TMP/in.bin" "$TMP/f.bin" "$TMP/out.bin" | awk '/time_ms/{print $2}')
        if ./compare "$TMP/ref.bin" "$TMP/out.bin" > "$TMP/cmp.txt"; then
          st=PASS; pass=$((pass + 1))
        else
          st=FAIL; fail=$((fail + 1)); cat "$TMP/cmp.txt"
        fi
        echo "$M,$N,$C,$K,$W,$name,$gt,$st" >> "$CSV"
        line="$line | $name ${gt}ms $st"
      done
      echo "$line"
    done
  done
done
echo "Done: $pass passed, $fail failed. Results in $CSV"
[ "$fail" -eq 0 ]
