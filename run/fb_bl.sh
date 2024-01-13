ALGO=gcn
for DATASTR in citeseer
do
    for SEED in 41 42 43
    do
        OUTDIR=./save/${DATASTR}/${ALGO}/${SEED}-0.0-0.0
        mkdir -p ${OUTDIR}
        OUTFILE=${OUTDIR}/out.txt
        python -u run_fb.py --seed ${SEED} --config ./config/${DATASTR}-bl.json --dev ${1:--1} \
            --thr_a 0.0 --thr_w 0.0 >> ${OUTFILE} &
        echo $! && wait
    done
done
