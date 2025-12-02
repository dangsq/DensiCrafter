
# DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures

The rapid progress of 3D generative models has enabled automatic synthesis of geometry and texture from multimodal inputs (e.g., text or images). However, most existing methods overlook physical constraints and manufacturability, resulting in designs that are difficult to fabricate or structurally unsound.

**DensiCrafter** addresses this gap by generating lightweight, self-supporting 3D hollow structures through **density-field optimization**. Starting from coarse voxel grids produced by TRELLIS, we reinterpret the grids as a continuous density field and introduce three differentiable, physically constrained, simulation-free loss terms. A mass regularization term penalizes unnecessary material while a restricted optimization domain preserves the object’s exterior geometry.

Our framework integrates seamlessly with pretrained TRELLIS-based 3D generative models (e.g., TRELLIS, DSO) **without architectural modifications**. Experiments demonstrate up to **43% reduction in material mass** for text-to-3D generation while maintaining high geometric fidelity and improving structural stability. Real-world 3D-printing tests further validate that the produced hollow structures are manufacturable and self-supporting.

---

## TODO Status

* **Main Framework ✔**
* **Dataset ✔**
* **Evaluation Code (Coming Soon)**

---

## Dataset

The **validation / test dataset** can be downloaded from:

🔗 **Baidu Drive:** [https://pan.baidu.com/s/1Wr06oLWo7z7-Ba9h91Zcew?pwd=jtbw](https://pan.baidu.com/s/1Wr06oLWo7z7-Ba9h91Zcew?pwd=jtbw)

The validation set index file is located at:

```
./data/val_index
```

---

## Installation & Setup

### Install TRELLIS

Please follow the official installation instructions from the
🔗 **[TRELLIS Repository](https://github.com/microsoft/TRELLIS)**.

## Run

To reproduce our examples for **multi-view**, **single-view**, and **text-to-3D** generation, simply run:

```bash
python main.py
```

Below is the core workflow (you may modify it to fit your own data):

```python
root_path = "./pretrained-models/"

# ---------------------------
# Image-to-3D Pipeline
# ---------------------------
pipeline_image = TrellisImageTo3DPipeline.from_pretrained(
    root_path + "TRELLIS-image-large"
)
pipeline_image.cuda()

image_paths = [
    "./test_200_new/1/1_1.png",
    "./test_200_new/1/1_2.png",
    "./test_200_new/1/1_3.png",
]

run_multi_image_task(pipeline_image, image_paths, "results/multi-image/1")
run_single_image_task(pipeline_image, image_paths[0], "results/single-image/1")

# ---------------------------
# Text-to-3D Pipeline
# ---------------------------
pipeline_text = TrellisTextTo3DPipeline.from_pretrained(
    root_path + "TRELLIS-text-xlarge"
)
pipeline_text.cuda()

prompt = (
    "A stylized cartoon cat character with a spherical head, pointed ears, "
    "prominent eyes, a curved mouth, a cylindrical torso with a white patch, "
    "elongated arms with large hands, tapered legs, and rounded feet. "
    "Predominantly black with white details."
)

run_text_task(pipeline_text, prompt, "results/text/1")
```

---

## Citation

If you find this repository helpful for your research, please consider citing:

```
@article{dang2025densicrafter,
  title={DensiCrafter: Physically-Constrained Generation and Fabrication of Self-Supporting Hollow Structures},
  author={Dang, Shengqi and Chai, Fu and Li, Jiaxin and Yuan, Chao and Ye, Wei and Cao, Nan},
  journal={arXiv preprint arXiv:2511.09298},
  year={2025}
}
```
