# Generative AI and Its Applications (ETMMGA304)

Implementations of every laboratory assignment and the capstone from the
**K.R. Mangalam University MCA laboratory manual for ETMMGA304 – Generative AI
and Its Applications**. Each lab is a standalone Python script with a matching
Google Colab notebook.

| Lab | Topic | Script | Notebook | Hardware |
|-----|-------|--------|----------|----------|
| 1 | Variational Autoencoder on MNIST | [lab1_vae_mnist.py](lab1_vae_mnist.py) | [notebooks/lab1_vae_mnist.ipynb](notebooks/lab1_vae_mnist.ipynb) | CPU OK (~2–5 min) |
| 2 | GAN on Fashion-MNIST | [lab2_gan_fashion_mnist.py](lab2_gan_fashion_mnist.py) | [notebooks/lab2_gan_fashion_mnist.ipynb](notebooks/lab2_gan_fashion_mnist.ipynb) | CPU OK (~10 min), GPU faster |
| 3 | GPT-2 decoding and fine-tuning | [lab3_gpt2_text_generation.py](lab3_gpt2_text_generation.py) | [notebooks/lab3_gpt2_text_generation.ipynb](notebooks/lab3_gpt2_text_generation.ipynb) | CPU OK |
| 4 | Stable Diffusion prompt-to-image | [lab4_stable_diffusion.py](lab4_stable_diffusion.py) | [notebooks/lab4_stable_diffusion.ipynb](notebooks/lab4_stable_diffusion.ipynb) | **CUDA GPU required** |
| Capstone | Study Tip Generator (Streamlit) | [capstone_app.py](capstone_app.py) | – | CPU OK |

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md).

## Course outcomes

- **CO1** Explain the theoretical foundations of VAEs, GANs and Transformers.
- **CO2** Implement generative models with open-source tools to generate text, images or other data.
- **CO3** Evaluate and compare generated outputs using quantitative and visual measures.
- **CO4** Analyse the ethical, social and practical implications of real-world deployment.

## Repository layout

```
GenAI/
├── README.md
├── LAB_RECORD.md                  # observation tables + viva answers
├── requirements.txt
├── lab1_vae_mnist.py
├── lab2_gan_fashion_mnist.py
├── lab3_gpt2_text_generation.py
├── lab4_stable_diffusion.py
├── capstone_app.py                # Streamlit demo (uses Lab 3 model)
├── notebooks/                     # Colab versions of Labs 1–4
├── data/                          # created automatically (datasets)
└── lab_outputs/                   # created automatically (images, checkpoints)
```

`data/` and `lab_outputs/` are created on first run and ignored by git.

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI.git
cd GenAI
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### CPU or CUDA GPU

- `pip install -r requirements.txt` installs the default PyTorch wheel. On
  Linux that wheel includes CUDA support. On Windows it is CPU-only.
- For an NVIDIA GPU on Windows, install the CUDA build first, then the rest
  (pick the CUDA version that matches your driver on
  [pytorch.org](https://pytorch.org/get-started/locally/)):

  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install -r requirements.txt
  ```
- Check which device PyTorch sees:
  `python -c "import torch; print(torch.cuda.is_available())"`

Labs 1–3 and the capstone run on CPU. **Lab 4 needs a CUDA GPU** (about 4 GB
VRAM in float16). The script stops with a message if no GPU is found. Use a
Colab T4 runtime if you do not have a local GPU.

The first run of each lab downloads its dataset or model weights (MNIST and
Fashion-MNIST are about 30 MB each, GPT-2 about 550 MB, Stable Diffusion v1.5
about 4 GB), so you need an internet connection.

## Running the labs

Run every command from the repository root.

```bash
# Lab 1: VAE on MNIST (5 epochs)
python lab1_vae_mnist.py
#   -> lab_outputs/vae_mnist.pt, vae_reconstructed.png, vae_generated.png, vae_latent.png

# Lab 2: GAN on Fashion-MNIST (10 epochs)
python lab2_gan_fashion_mnist.py
#   -> lab_outputs/gan_epoch_01.png, gan_epoch_05.png, gan_epoch_10.png,
#      gan_losses.png, fashion_generator.pt

# Lab 3: GPT-2 decoding comparison, fine-tuning and held-out comparison
python lab3_gpt2_text_generation.py
#   -> lab_outputs/study_tip_gpt2/, lab3_decoding.csv, lab3_held_out.csv

# Lab 4: Stable Diffusion (GPU only; run in a fresh session)
python lab4_stable_diffusion.py
#   -> lab_outputs/library_prompt_{1,2}_seed_{42,43}.png

# Capstone: Streamlit demo (run Lab 3 first)
streamlit run capstone_app.py
```

Optional flags: `--epochs N` and `--batch-size N` (Labs 1–2), and
`--epochs N` and `--skip-decoding` (Lab 3). Use `python <script> --help` to
list them.

## Google Colab

1. Open a notebook from `notebooks/` in Colab (File → Upload notebook, or
   *Open from GitHub* with this repository's URL).
2. For Labs 2 and 4, choose **Runtime → Change runtime type → T4 GPU**.
3. Run the cells in order. The first cell installs the packages that lab needs.
4. Download `lab_outputs/` before the session ends.

## Reproducibility

Every script uses `SEED = 42`, prints the device it runs on, and writes its
outputs to `lab_outputs/`. Results still vary a little between CPU and GPU and
between library versions. Record your own losses and samples in
[LAB_RECORD.md](LAB_RECORD.md); do not copy the expected ranges given there.

## Troubleshooting

| Symptom | What to check |
|---------|---------------|
| CUDA unavailable | Select a GPU runtime, or install the CUDA PyTorch build. CPU is fine for Labs 1–3. |
| GPU out of memory | Close other models, reduce `--batch-size`, and run Lab 4 in a fresh session. |
| Dataset or model download fails | Check your connection and Hugging Face access, then rerun. |
| Capstone says the model is missing | Run `python lab3_gpt2_text_generation.py` so that `lab_outputs/study_tip_gpt2/` exists. |

## References

- [Torchvision MNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
- [Torchvision FashionMNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html)
- [Hugging Face Transformers text generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Stable Diffusion v1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [Stable Diffusion on Replicate](https://replicate.com/stability-ai/stable-diffusion/readme)
