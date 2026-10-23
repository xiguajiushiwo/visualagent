# RAM visual QC minimal stack

This is a barebones pipeline for:

1. subject localization
2. overlapping tile scan
3. Zhipu VLM inspection
4. local recheck on suspicious areas
5. optional SAM box refinement

## Run

```bash
python ram_scan.py path/to/image.png --api-key $ZHIPU_API_KEY --outdir runs/demo
```

## Manual subject box

If you already know the RAM region:

```bash
python ram_scan.py path/to/image.png --subject-bbox 1100,2900,1900,3648 --api-key $ZHIPU_API_KEY
```

## Optional SAM

If you want SAM refinement, install the official package and provide a checkpoint:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then set:

```bash
python ram_scan.py path/to/image.png --sam-checkpoint path/to/sam_vit_h_4b8939.pth
```

If `torch` / `segment_anything` is missing, the script still runs and falls back to box-only refinement.

## Outputs

- `results.json`
- `annotated.png`

