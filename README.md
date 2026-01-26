# Pixel-Level Foreground Perception with Soft Consistency Guidance for Visible-Infrared Person Re-Identification

<!-- You need to go to Zenodo.org, link this repo, and paste the DOI badge markdown here. It looks like this: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxxx.svg)](https://doi.org/10.5281/zenodo.xxxxxx) -->
<!-- PLACEHOLDER FOR ZENODO DOI BADGE -->

> **Note to Readers and Reviewers:**  
> This code is the official implementation of the manuscript submitted to **The Visual Computer**, titled **"Pixel-Level Foreground Perception with Soft Consistency Guidance for Visible-Infrared Person Re-Identification"**. The source code is permanently hosted to ensure transparency and reproducibility of the experiments presented in the paper.

## 1. Introduction

Visible-Infrared Person Re-Identification (VI-ReID) is challenged by significant modality discrepancy inherent in background regions. To address this, we propose the **Soft Foreground Guided Perception Network (SFGPN)**, which utilizes a Foreground Region Localization (FRL) module and Soft Consistency Guidance (SCG) to explicitly enhance foreground representations at the pixel level while maintaining robustness against segmentation noise.

This repository contains the full implementation, including data preprocessing, training, and testing scripts.

## 2. Model Architecture & Key Algorithms

To facilitate the replication of our methods, the implementations of key components are located as follows:

- **FRL Module & CSFP Module**: Implemented in `model.py` 
- **Soft Consistency Guidance (SCG) Loss**: Defined in `loss.py`
- **Data Augmentation (BGS/BRE)**: Implemented in `data_loader.py` 

## 3. Results

Performance comparison on the SYSU-MM01 dataset:

| Datasets  | Rank@1 | Rank@10 | Rank@20 |  mAP   |                        Trained Model                         | Processed Data (Instance Segmented)                          |
| :-------: | :----: | :-----: | :-----: | :----: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| SYSU-MM01 | 75.33% | 97.17%  | 99.16%  | 72.26% | [GoogleDrive](https://drive.google.com/drive/folders/1guJetD4OFoohCVma5rUBq2TK9Yxhuv1F?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1VCFskf1Kx0rANla0l9tD1g?pwd=1234) | [Baidu Netdisk](https://pan.baidu.com/s/1Pw22313pqSaBGebggMH-uQ?pwd=1234) |

## 4. Prerequisites

### Environment
Please refer to `SFGPN/requirements.txt` for the specific dependency versions.
```bash
pip install -r SFGPN/requirements.txt
```

### Datasets

You need to download the original datasets and then perform the instance segmentation preprocessing.

1.  **Original Datasets Download:**
    -   **RegDB [1]:** Download from [here](http://dm.dongguk.edu/link.html).
    -   **SYSU-MM01 [2]:** Download from [here](http://isee.sysu.edu.cn/project/RGBIRReID.htm).
    -   **LLCM [3]:** Download by sending a signed [agreement](https://github.com/ZYK100/LLCM/blob/main/Agreement/LLCM%20DATASET%20RELEASE%20AGREEMENT.pdf) to `zhangyk@stu.xmu.edu.cn`.

2.  **Instance Segmentation Processing:**
    To obtain the mask annotations, we use the **YOLO11** instance segmentation model.
    -   **Logic:** Refer to `SFGPN/PedestrianSegmentation/Segmentation.py`.
    -   **Model:** Get the YOLO11 model from [Ultralytics](https://docs.ultralytics.com/zh/tasks/segment/).
    
    > **Note on Dirty Data:** The original datasets contain "dirty" images (background only, no pedestrians). For example, in SYSU-MM01, we set a confidence threshold of 0.1. Images below this threshold are saved to `SYSU-MM01-dirty` (e.g., subfolders like `cam1`,`cam2`, `cam5/0461` often contain empty backgrounds). We filter these out during preprocessing.

3.  **Quick Start (Pre-processed Data):**
    If you prefer not to process the data manually, you can download our **processed datasets** directly from [Baidu Netdisk](https://pan.baidu.com/s/1Pw22313pqSaBGebggMH-uQ?pwd=1234).

## 5. Training

### SYSU-MM01
First, run `pre_process_sysu_mask.py` to generate the `SYSU-MM01-npy` folder.
```bash
python train.py --dataset sysu --gpu 0 --workers 4
```

### LLCM
```bash
python train.py --dataset llcm --gpu 0 --workers 4
```

### RegDB
```bash
python train.py --dataset regdb --gpu 0 --workers 4 --trial 1
```

**Arguments:**
- `--dataset`: Choose from "sysu", "regdb", or "llcm".
- `--gpu`: Specify the GPU ID.

*Pre-trained weights are available on [GoogleDrive](https://drive.google.com/drive/folders/1guJetD4OFoohCVma5rUBq2TK9Yxhuv1F?usp=drive_link) and [Baidu Netdisk](https://pan.baidu.com/s/1VCFskf1Kx0rANla0l9tD1g?pwd=1234).*

## 6. Testing

### SYSU-MM01
```bash
python test_matrix.py --dataset 'sysu' --mode 'all' --resume 'sysu_agw_p4_n6_lr_0.1_seed_0_best.t'  --gpu 0 --workers 4
```

`--mode`: "all" (All Search) or "indoor" (Indoor Search).

### LLCM
*Note: Please modify the `test_mode` variable in the script or configuration before running.*

**VIS to IR mode:**
Set `test_mode = [2, 1]` in the code.
```bash
python test_matrix.py --dataset 'llcm' --resume 'llcm_agw_p4_n6_lr_0.1_seed_0_best.t'  --gpu 0 --workers 4
```

**IR to VIS mode:**
Set `test_mode = [1, 2]` in the code.
```bash
python test_matrix.py --dataset 'llcm' --resume 'llcm_agw_p4_n6_lr_0.1_seed_0_best.t'  --gpu 0 --workers 4
```

### RegDB
*Note: For RegDB, modify the data loading logic in the script as follows:*

**VIS to IR mode:**
```python
test_mode = [2, 1]
query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
```
Run:
```bash
python test_matrix.py --dataset 'regdb' --gpu 0
```

**IR to VIS mode:**
```python
test_mode = [1, 2]
query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
```
Run:
```bash
python test_matrix.py --dataset 'regdb' --gpu 0
```

## 7. Citation

If this work and code are helpful for your research, please cite our paper. **We strongly encourage readers to refer to the finalized version published in *The Visual Computer*:**

```bibtex
@article{Wu2026SFGPN,
  title={Pixel-Level Foreground Perception with Soft Consistency Guidance for Visible-Infrared Person Re-Identification},
  author={Wu, Shuzuan and Zhang, Chengfang and Feng, Ziliang},
  journal={The Visual Computer},
  year={2026}
}
```

## 8. Acknowledgments & References

Most of the code is based on [DEEN](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) [3]. We thank the authors for their contributions to the community.

[1] D. T. Nguyen et al. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors,17(3):605,2017.

[2] A. Wu et al. RGB-infrared cross-modality person re-identification. ICCV, 2017.

[3] Zhang Y, Wang H. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification.CVPR,2023.
