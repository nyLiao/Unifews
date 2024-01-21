ALGO=sgc
for DATASTR in arxiv
do
    for SEED in 41 42 43
    do
        OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-1e-08-0.0
        mkdir -p ${OUTDIR}
        OUTFILE=${OUTDIR}/out.txt
        python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
            --algo ${ALGO} >> ${OUTFILE} &
        echo $! && wait
    done
done
