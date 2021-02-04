python -m hypothesis.bin.ratio_estimation.train \
    --batch-size 512 \
    --epochs 10 \
    --lr 0.0001 \
    --show \
    --lrsched-on-plateau \
    --conservativeness 0.0 \
    --data-test "ratio_estimation.DatasetTest" \
    --data-train "ratio_estimation.DatasetTrain" \
    --estimator "ratio_estimation.RatioEstimator"
