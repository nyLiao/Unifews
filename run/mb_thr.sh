ALGO=sgc_thr
for DATASTR in products
do
    for THRA in 2.0e-07 5.0e-07 1.0e-06
    # for THRA in 5e-05 1e-04 2e-04
    do
        for THRW in 1.0e-03 5.0e-03 1.0e-02
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
