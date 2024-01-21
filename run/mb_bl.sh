ALGO=sgc
for DATASTR in products
do
    for SEED in 41 42 43
    do
        OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-1.0e-08-0.0e+00
        mkdir -p ${OUTDIR}
        OUTFILE=${OUTDIR}/out.txt
        python -u run_mb.py --seed ${SEED} --config ./config/${DATASTR}_mb.json --dev ${1:--1} \
            --algo ${ALGO} >> ${OUTFILE} &
        echo $! && wait
    done
done
