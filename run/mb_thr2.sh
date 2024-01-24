ALGO=sgc_thr
for DATASTR in arxiv
do
    for THRA in 2.0e-06 5.0e-06 1.0e-05
    do
        for THRW in 1.0e-02 1.0e-01
        do
            SEED=26
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} &
            echo $! && wait
        done
    done
done
