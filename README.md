# KPI-NeRF

Official implementation of **KPI-NeRF: Hyperspectral Neural Radiance Fields from a Single Kaleidoscopic Plenoptic Image**.

KPI-NeRF reconstructs a view- and spectrum-dependent radiance field from a single snapshot hyperspectral kaleidoscopic plenoptic image. It uses an implicit neural representation with a learned spectral subspace to interpolate light-field views and demultiplex up to 29 spectral channels.

## Paper

- **Title:** KPI-NeRF: Hyperspectral Neural Radiance Fields from a Single Kaleidoscopic Plenoptic Image
- **Authors:** Erqi Huang, John Restrepo, Xun Cao, Ivo Ihrke
- **Venue:** Computer Graphics Forum, 2026
- **DOI:** [10.1111/cgf.70590](https://doi.org/10.1111/cgf.70590)
- **Project repository:** [github.com/erqihuang/MLF-Nerf](https://github.com/erqihuang/MLF-Nerf)

## Results and media

The repository includes an EPI visualization preview:

[Open the EPI visualization PDF](assets/7-EPI.pdf)

Place additional representative figures, rendered novel views, comparison videos, or GIFs in `assets/`. The repository intentionally does not include the local research figures or logs.

## Environment

The code is intended for Linux with an NVIDIA GPU and CUDA-compatible PyTorch. Python 3.8 is specified in `environment.yml`; exact GPU memory requirements depend on image resolution, spectral channels, and sample count.

```bash
conda env create -f environment.yml
conda activate kpi-nerf
```

The environment includes PyTorch, torchvision, NumPy, SciPy, OpenCV, ImageIO, Matplotlib, TensorBoard, LPIPS, scikit-image, and Open3D. Open3D is only needed by visualization-related code and may require additional system display support.

## Data

Raw data is not distributed in this code repository. Upload the data to a separate storage location and add the download link here:

> **Dataset download:** TODO: add the public dataset URL.

The scripts expect scene data below `data/`, for example:

```text
data/
├── <date>/<scene-name>/       # scene images, calibration and metadata
└── GT/<scene-name>.csv        # ground-truth values for evaluation
```

Update `SCENE_NAME`, ground-truth paths, and calibration-related arguments in the example scripts to match the downloaded data. The exact file contents are defined by `SpecDataLoaderAnyFolder` in the retained task code.

## Quick start

### Training and evaluation: ajar scene

Edit `SCENE_NAME` and `TRAIN_GPU_ID` in the script, then run:

```bash
bash config/ajar_kpi-nerf.sh
```

This trains `train_gt.py`, writes a log under `logs/`, resolves the newest matching checkpoint, and runs `spiral_gt.py` for novel-view evaluation.

### Ball-scene evaluation example

Copy the trained checkpoint and dataset into the paths configured in `config/ball_kpi-nerf.sh`, then run:

```bash
GPU_ID=0 bash config/ball_kpi-nerf.sh
```

The script is a cleaned, shareable version of the original `ball.sh` command. It runs `spiral_Tcomp.py` and supports the 29-channel example configuration.

### Direct Python entry points

```bash
python tasks/any_folder_spec_klensPlus_RGB_coding/train_gt.py --help
python tasks/any_folder_spec_klensPlus_RGB_coding/spiral_gt.py --help
python tasks/any_folder_spec_klensPlus_RGB_coding/spiral_Tcomp.py --help
```

## Important parameters

- `--scene_name`: scene directory relative to `--base_dir`.
- `--gpu_id`: CUDA device used by the process.
- `--nerf_lr`: NeRF learning rate.
- `--epoch`: number of training epochs.
- `--num_sample`: samples along each ray.
- `--spec_chnls`: reconstructed spectral channel count; the paper examples use 29.
- `--bf_num` and `--init_basis`: spectral basis size and initialization.
- `--ls_factor`: light-source power factor used by the data model.
- `--resize_ratio`: image downsampling ratio.
- `--learn_t`, `--learn_R`, `--learn_focal`: enable optimization of camera translation, rotation, and focal parameters.

For the complete argument list, use the `--help` commands above and inspect the parser definitions in the task scripts.

## Repository layout

```text
├── config/                              # reproducible example shell scripts
│   ├── ajar_kpi-nerf.sh                 # training + evaluation example
│   └── ball_kpi-nerf.sh                  # ball-scene evaluation example
├── dataloader/                          # input and calibration loaders
├── models/                              # MLF-NeRF and camera parameter models
├── tasks/any_folder_spec_klensPlus_RGB_coding/
│                                         # retained training and rendering entry points
├── third_party/                         # third-party source used by the project
├── utils/                               # rendering, geometry and evaluation utilities
├── environment.yml                      # Conda environment
└── README.md
```

## Evaluation

The retained rendering/evaluation scripts generate novel views and spectral outputs. Quantitative values should be reported together with the dataset split, scene, resolution, spectral-channel count, and checkpoint. A results table will be added after the public data and reproducible checkpoints are released.

## Citation

```bibtex
@inproceedings{huang2026kpi,
  title={KPI-NeRF: Hyperspectral Neural Radiance Fields from a Single Kaleidoscopic Plenoptic Image},
  author={Huang, Erqi and Restrepo, John and Cao, Xun and Ihrke, Ivo},
  booktitle={Computer Graphics Forum},
  pages={e70590},
  year={2026},
  organization={Wiley Online Library}
}
```

## Acknowledgements

This project builds on ideas and code from NeRF-style neural rendering and the preceding MLF/NeRF implementation. Please retain the original notices in third-party components.

## License

No license has been selected for this repository yet. Before publishing, add a `LICENSE` file and update this section with the chosen license and any third-party license obligations.
