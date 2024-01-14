ALGO=gcn_thr
for DATASTR in cs
do
    # for THRA in 0.3 0.5 0.7 0.9 1.2 1.5 1.8
    for THRA in 0.3 0.7 1.2 1.5 1.8
    do
        for THRW in 0.3 0.7 1.2 1.5 1.8
        do
            SEED=41
            OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-${THRA}-${THRW}
            mkdir -p ${OUTDIR}
            OUTFILE=${OUTDIR}/out.txt
            python -u run_fb.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} \
                --algo ${ALGO} --thr_a ${THRA} --thr_w ${THRW} >> ${OUTFILE} &
            echo $! && wait
        done
    done
done
