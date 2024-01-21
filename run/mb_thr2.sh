ALGO=sgc_thr
for DATASTR in cora
do
    for THRA in 5.0e-05 1.0e-04 2.0e-04
    do
        for THRW in 0.01 0.05
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
