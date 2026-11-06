set -e


mkdir -p results
IMG=${1:-}
if [ -z "$IMG" ]; then ./imgtool synth 4096 3072 results/photo.ppm; IMG=results/photo.ppm; fi
C=$(./imgtool img2tensor "$IMG" results/input.bin | sed -n 's/.*C=\([0-9]*\).*/\1/p')

tm() { grep time_ms | awk '{print $2}'; }
NP=$(nproc)
SUM=results/summary.txt
echo "image: $IMG   C=$C   cores: $NP" | tee $SUM
printf "%-8s %-11s %12s %9s\n" bank version time_ms speedup | tee -a $SUM

for b in edges8 gauss3 gauss5 gauss7 sharpen; do
    ./imgtool filters $b $C results/f_$b.bin > results/f_$b.txt
    base=$(./cpu_reference results/input.bin results/f_$b.bin results/$b.naive.bin naive | tm)
    printf "%-8s %-11s %12.1f %9s\n" $b cpu_naive $base 1.0x | tee -a $SUM
    t=1
    while [ $t -le $NP ]; do
        ms=$(OMP_NUM_THREADS=$t ./cpu_reference results/input.bin results/f_$b.bin results/$b.fast.bin | tm)
        printf "%-8s %-11s %12.1f %8.1fx\n" $b "cpu_fast_t$t" $ms $(awk "BEGIN{print $base/$ms}") | tee -a $SUM
        [ $t -lt $NP ] && [ $((t*2)) -gt $NP ] && t=$NP || t=$((t*2))
    done
    ./imgtool compare results/$b.naive.bin results/$b.fast.bin | sed "s/^/         fast vs naive: /" | tee -a $SUM
    for g in conv_naive conv_opt; do        # GPU versions, if built (same command-line form assumed)
        if [ -x ./$g ]; then
            ms=$(./$g results/input.bin results/f_$b.bin results/$b.$g.bin | tm)
            printf "%-8s %-11s %12.1f %8.1fx\n" $b $g $ms $(awk "BEGIN{print $base/$ms}") | tee -a $SUM
            ./imgtool compare results/$b.naive.bin results/$b.$g.bin | sed "s/^/         $g vs naive: /" | tee -a $SUM
        fi
    done
    if [ $b = edges8 ]; then ./imgtool tensor2img results/$b.fast.bin results/edges8 ccccaaac > /dev/null
    elif [ "$C" = 3 ]; then ./imgtool tensor2rgb results/$b.fast.bin results/$b.ppm > /dev/null
    else ./imgtool tensor2img results/$b.fast.bin results/$b > /dev/null; fi
    rm -f results/$b.*.bin
done
echo "edges8 images: results/edges8_0..7.pgm = identity box_blur gaussian sharpen sobel_x sobel_y laplacian emboss" | tee -a $SUM
