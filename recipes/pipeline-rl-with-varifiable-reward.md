## Load dataset
```
cd pipelie-rl
mkdir datasets
python scripts/load_dataset.py --hf_dataset hf-imo-colab/olympiads-ref-base-exact-matching --save_dir tmp/datasets --split train
python scripts/load_dataset.py --hf_dataset hf-imo-colab/olympiads-ref-base-exact-matching --save_dir tmp/datasets --split test
```

## Run experiment

```
scripts/run_imo_exact_matching.sh
```