# Run F — D=128 Capacity Test

## Config
D=128 DFF=256 H=4 L=3 lr=1e-3 N=6000 B=8 NVAL=256

## Results
- Train final (steps 5401-6000): 0.7953
- **VAL mean = 0.8058** (std=0.3111, sem=0.0194, n=256)
- Perplexity: 2.24

## Comparison
| Run | Params | Val | vs Bigram |
|-----|--------|-----|-----------|
| Bigram | — | 1.314 | baseline |
| Run E | 118K | 0.8544 | -0.459 |
| **Run F** | **433K** | **0.8058** | **-0.508** |

## Verdict
- Val 0.806 << 0.854 (Run E): capacity adds 5.7% improvement
- Train 0.795 vs Val 0.806: NO overfitting (gap 0.01)
- Under-trained: 1.54M tokens / 433K params ≈ 3.5 tok/param
- Chinchilla-optimal ≈ 8.6M tokens → need 5.6x more

## Next
N=12000 (double steps) → expect ~0.78-0.79
Then N=24000 if still improving
