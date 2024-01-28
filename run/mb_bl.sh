for DATASTR in papers
do
    for ALGO in gbp gbp_agp
    do
        for SEED in 41
        do
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-1.0e-08-0.0e+00
            mkdir -p ${OUTDIR}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a 1.0e-08 --thr_w 0.0 >> ${OUTFILE} &
            echo $! && wait
        done
    done
done
