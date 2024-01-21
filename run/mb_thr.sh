ALGO=sgc_thr
for DATASTR in arxiv
do
    for THRA in 1e-08 2e-08 5e-08 1e-07 2e-07
    # for THRA in 5e-05 1e-04 2e-04
    do
        for THRW in 1e-05 2e-05 5e-05 1e-04 2e-04 5e-04
        # for THRW in 0.001 0.002 0.005 0.01 0.02 0.05
        # for THRW in 0.01 0.05
        do
            SEED=43
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
            mkdir -p ${OUTDIR}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
            echo $! && wait
        done
    done
done
