ALGO=gcn
for DATASTR in cs
do
    for SEED in 41 42 43
    do
        OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-0.0-0.0
        mkdir -p ${OUTDIR}
        OUTFILE=${OUTDIR}/out.txt
        python -u run_fb.py --seed ${SEED} --config ./config/${DATASTR}.json --dev ${1:--1} \
            --algo ${ALGO} >> ${OUTFILE} &
        echo $! && wait
    done
done
