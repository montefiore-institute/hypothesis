python -m hypothesis.bin.ratio_estimation.train \
    --show \
    --batch-size 1024 \
    --epochs 25 \
    --lr 0.01 \
    --lrsched-on-plateau \
    --conservativeness 0.0 \
    --data-test "ratio_estimation.DatasetTest" \
    --data-train "ratio_estimation.DatasetTrain" \
    --estimator "ratio_estimation.RatioEstimator"
