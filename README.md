# CLIPDet3D

Official implementation of **CLIPDet3D**, a camera-only 3D object detector for
nuScenes that combines CLIP visual-language features, geometry-aware BEV
lifting, temporal fusion, foreground prompting, and optional monocular-depth
feature distillation.

The released source has been checked against the published R50 and R101 EMA
checkpoints. Every model parameter name and tensor shape matches the two
checkpoints exactly.

## Results on nuScenes validation

| Model | Backbone | Image size | Batch / GPU | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIPDet3D | ResNet-50 | 256 x 704 | 8 | 0.462 | 0.566 | 0.507 | 0.266 | 0.405 | 0.252 | 0.206 |
| CLIPDet3D | ResNet-101 | 512 x 1408 | 2 | 0.467 | 0.573 | 0.513 | 0.252 | 0.328 | 0.300 | 0.208 |

The archived R101 evaluation file gives mAP 0.466565, NDS 0.572827, mATE
0.513331, mASE 0.254015, mAOE 0.328672, mAVE 0.300265, and mAAE 0.208270.
The small mASE difference from the rounded paper table is retained here for
transparency.

## Model weights

Download the released EMA checkpoints from the
[CUMT CloudSpace folder](https://pan.cumt.edu.cn/share/44e3c533e59146c9a319a8251d):

- `CLIPDet3D_R50_256x704_epoch10_ema.pth`
- `CLIPDet3D_R101_512x1408_epoch20_ema.pth`

SHA-256 checksums:

```text
c8a878ef1d5f71d50c226189d853b75acc0f0873ca522bc0575ed66766a2266e  CLIPDet3D_R50_256x704_epoch10_ema.pth
96e4d36f6b987f22733d70c4f4603c628eb2011154c83e4070d2da52e1e745f8  CLIPDet3D_R101_512x1408_epoch20_ema.pth
```

Place the downloaded model weights in `work_dirs/` or another path of your
choice. The R50 training configuration also uses the Depth Anything V2 ViT-S
teacher checkpoint at `ckpts/depth_anything_v2_vits.pth`.

## Environment

The checkpoint-compatibility test was run with:

- Python 3.8
- PyTorch 2.0.0 + CUDA 11.8
- torchvision 0.15.0
- MMCV 1.6.0
- MMDetection 2.28.2
- MMSegmentation 0.30.0
- MMDetection3D 1.0.0rc4

Install the OpenMMLab dependencies following their CUDA-specific
instructions, then install this repository:

```bash
pip install -r requirements.txt
pip install timm opencv-python
pip install -v -e .
```

The CLIP backbones are expected at:

```text
ckpts/RN50.pt
ckpts/RN101.pt
```

You may change `ckpt_path` in the relevant config if the files are stored
elsewhere.

## Data preparation

Download nuScenes v1.0-trainval and nuScenes lidarseg annotations, then arrange
the data as follows:

```text
CLIPDet3D/
  data/nuscenes/
    maps/
    samples/
    sweeps/
    v1.0-trainval/
    lidarseg/
```

Generate the metadata and image supervision files using the provided tools:

```bash
python tools/create_data_bevdet.py
python tools/process_class_mask.py
```

The released configs expect `geobev-nuscenes_infos_train.pkl` and
`geobev-nuscenes_infos_val.pkl` under `data/nuscenes/`.

## Training

R50, 256 x 704, long-term temporal model with depth-feature distillation:

```bash
bash tools/dist_train.sh \
  configs/clipdet3d/clipdet3d-r50-cbgs-longterm.py 4
```

R101, 512 x 1408:

```bash
bash tools/dist_train.sh \
  configs/clipdet3d/clipdet3d-r101-cbgs.py 4
```

Adjust the number of processes and `samples_per_gpu` to fit your hardware.

## Evaluation

```bash
# R50
bash tools/dist_test.sh \
  configs/clipdet3d/clipdet3d-r50-cbgs-longterm.py \
  /path/to/CLIPDet3D_R50_256x704_epoch10_ema.pth \
  4 --eval bbox

# R101
bash tools/dist_test.sh \
  configs/clipdet3d/clipdet3d-r101-cbgs.py \
  /path/to/CLIPDet3D_R101_512x1408_epoch20_ema.pth \
  4 --eval bbox
```

## Released configurations

- `configs/clipdet3d/clipdet3d-r50-cbgs-longterm.py`: 256 x 704, eight-frame
  temporal setting, R50 CLIP backbone, Depth Anything V2 distillation.
- `configs/clipdet3d/clipdet3d-r101-cbgs.py`: 512 x 1408, two-frame temporal
  setting, R101 CLIP backbone, no depth teacher in the released checkpoint.

## Acknowledgements

This project is built on MMDetection3D and GeoBEV, and uses CLIP and Depth
Anything V2 components. We thank their authors and the open-source community.

## License

Please see [LICENSE](LICENSE).
