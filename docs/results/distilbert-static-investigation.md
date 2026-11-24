# DistilBERT static INT8 investigation

fp32 baseline on the same 300 SST-2 validation samples: 89.67% accuracy.

| Run | Config | Accuracy | Notes |
|---|---|---|---|
| Phase 4 | all supported ops, minmax | 50.92% (n=872) | Equals the positive-class rate: model predicts one class |
| A | MatMul only, minmax | 77.67% (n=300) | Speedup figure discarded: fp32 baseline ran at 375 ms (machine under load) vs ~136 ms normally |
| B | MatMul only, percentile | 85.33% (n=300) | 1.11x faster, 214 MB (16% smaller) |

Conclusion: static INT8 is unsuitable for this model at this calibration size. Dynamic INT8 gives
90.37% (n=872), 4x smaller and 1.46x faster, and is the recommended variant.