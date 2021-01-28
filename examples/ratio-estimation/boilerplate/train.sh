python -m hypothesis.bin.ratio_estimation.train \
    --show \
    --batch-size 2048 \
    --epochs 100 \
    --lr 0.001 \
    --conservativeness 0.00 \
    --data-test "ratio_estimation.DatasetTest" \
    --data-train "ratio_estimation.DatasetTrain" \
    --estimator "ratio_estimation.RatioEstimator" \
    --alpha 0.05
