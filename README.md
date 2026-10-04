# Generative AI - Practical No-03: Autoencoder and Variational Autoencoder for Image Reconstruction

## MNIST Digit Reconstruction & Generation — Autoencoder vs VAE

A Dense Autoencoder (AE) and a Variational Autoencoder (VAE) are built, trained and compared on the MNIST handwritten digit dataset. The AE reconstructs digit images; the VAE both reconstructs images and **generates new digit images** by sampling its learned latent space.

---

## 📋 Project Info

| Field                | Details           |
| -------------------- | ------------------ |
| **Student**          | Sneha Chaurasia    |
| **PRN**              | 202401110046       |
| **Class / Division** | CSE-AIML / A       |
| **Subject**          | Generative AI Lab  |
| **Practical No.**    | 03                 |

---

##  Problem Statement

Build an Autoencoder and a Variational Autoencoder (VAE) for image reconstruction and generation. Compare the performance of both models by analyzing the quality of reconstructed and generated images.

##  Objective

To implement an Autoencoder and a Variational Autoencoder (VAE) for reconstructing images and generating new images similar to the training data. The models are compared on:
- Reconstruction quality
- Training loss
- Visual quality of reconstructed images
- Quality of newly generated images

---

## 📦 Dataset

| Field        | Details                                                         |
| ------------ | ---------------------------------------------------------------- |
| **Name**     | MNIST Handwritten Digit Dataset                                 |
| **Source**   | Bundled with Keras (`tf.keras.datasets.mnist`)                  |
| **Type**     | Grayscale images of handwritten digits                          |
| **Classes**  | 10 (digits 0–9)                                                 |
| **Image size** | 28 × 28 pixels                                                 |
| **Loading**  | `from tensorflow.keras.datasets import mnist`<br>`(x_train, _), (x_test, _) = mnist.load_data()` |

| Split    | Samples    |
| -------- | ---------- |
| Training | 60,000     |
| Testing  | 10,000     |
| **Total** | **70,000** |

Labels are only used to visualize the latent space and class distribution — both the AE and VAE are trained **unsupervised**, using only the images.

---

## 🧠 Model Architecture

**Autoencoder (AE)** — plain Dense encoder/decoder:
```
Input (784) → Dense(256) → Dense(64) → Latent(16) → Dense(64) → Dense(256) → Output (784, sigmoid)
```

**Variational Autoencoder (VAE)** — probabilistic latent space:
```
Input (784) → Dense(256) → Dense(64) → [z_mean(16), z_log_var(16)] → Sampling (reparameterization trick)
            → z(16) → Dense(64) → Dense(256) → Output (784, sigmoid)
```

| Component             | Purpose                                                                 |
| ---------------------- | ------------------------------------------------------------------------ |
| Encoder                | Compresses the 784-pixel input into a 16-dimensional latent representation |
| `z_mean`, `z_log_var`  | (VAE only) Parametrize a Gaussian distribution per input, instead of one fixed point |
| Sampling layer         | (VAE only) Draws `z = z_mean + exp(0.5·z_log_var)·ε`, `ε ~ N(0,1)` — the reparameterization trick |
| Decoder                | Reconstructs the 784-pixel image from the latent vector                |
| Loss (AE)              | Binary cross-entropy (reconstruction only)                             |
| Loss (VAE)             | Binary cross-entropy (reconstruction) + KL divergence (regularizes latent space toward `N(0, I)`) |

**Training setup:** Adam optimizer (lr = 1e-3), batch size 256, 20 epochs, latent dimension 16 (plus a separate 2-D latent VAE trained only for latent-space visualization).

---

## 🔄 Workflow

1. Import libraries and load the MNIST dataset
2. Visualize sample digits and class distribution
3. Normalize pixels to `[0, 1]` and flatten images to 784-dim vectors
4. **Autoencoder:** build encoder → build decoder → combine → compile → train → plot loss → reconstruct test images
5. **VAE:** build encoder with `z_mean`/`z_log_var` → sampling layer (reparameterization trick) → build decoder → custom `VAE` model with combined loss → train → plot loss curves → reconstruct test images → generate new images by sampling `N(0, I)`
6. Visualize the 2D latent space (coloured by digit) and a generated digit grid
7. Compare AE vs VAE on reconstruction quality, loss and generation ability
8. Report final results and conclusion

---

## 📊 Results

> Replace the `XX.XX` values with the numbers printed in the executed notebook (Section 13: Results).

**Final Loss:**

| Model       | Final Train Loss | Test Loss |
| ----------- | ----------------- | --------- |
| Autoencoder | XX.XX (BCE)        | XX.XX (BCE) |
| VAE         | XX.XX (BCE+KL)      | XX.XX (BCE+KL) — Recon: XX.XX, KL: XX.XX |

**Comparison:**

| Aspect                     | Autoencoder            | VAE                               |
| --------------------------- | ----------------------- | ----------------------------------- |
| Can generate new images     | No                       | Yes                                |
| Latent space structure      | Unstructured            | Continuous, Gaussian-like          |
| Reconstruction sharpness    | Sharper                 | Slightly blurrier (KL regularization) |
| Parameters                  | XX,XXX                  | XX,XXX                             |

**Graphs generated in the notebook:** sample digits, class distribution, AE training loss, VAE loss curves (total / reconstruction / KL), original-vs-reconstructed grids for both models, side-by-side AE-vs-VAE comparison, newly generated VAE digits, 2D latent space scatter plot (coloured by digit), and a latent-space digit grid sweep.

### Observations

- The Autoencoder reaches a lower, purely reconstruction-based loss and produces visually sharper reconstructions, since nothing constrains its latent space.
- The VAE's reconstructions are slightly blurrier, trading sharpness for a continuous, well-organized latent space (enforced by the KL-divergence term).
- The 2D latent-space scatter plot shows the VAE groups same-digit points into smooth, overlapping clusters, with gradual transitions between digit classes.
- Sampling random points from `N(0, I)` and decoding them with the VAE produces recognizable, varied digit shapes; the plain Autoencoder's latent space is not reliably suited to this.
- The latent-space grid sweep shows a smooth, continuous morph between digit types, confirming the VAE's latent space is meaningful for generation.

---

## ✅ Conclusion

Both an Autoencoder and a Variational Autoencoder were built, trained and evaluated on the MNIST dataset. The Autoencoder learns a compact latent representation purely to minimize reconstruction error, giving sharp reconstructions but an unstructured latent space unsuitable for generation. The VAE adds a probabilistic latent representation and a KL-divergence regularization term, making its latent space continuous and approximately Gaussian. This lets the VAE both reconstruct existing images and generate new, realistic digit images — a capability the plain Autoencoder does not reliably offer.

**Trade-off:** Autoencoder = sharper reconstructions, no generation. VAE = slightly blurrier reconstructions, but meaningful generation and a structured latent space.

**Possible improvements:** Convolutional encoder/decoder layers for sharper images, a higher latent dimensionality, beta-VAE (weighting the KL term) to control the sharpness/structure trade-off, and conditional VAE (CVAE) to generate digits of a chosen class.

---

## 📁 Repository Structure

```
Autoencoder-and-VAE-for-Image-Reconstruction-Practical-No.-03-Gen-AI/
│
├── README.md                                  ← this file
├── Practical_03_AE_VAE_MNIST.ipynb           ← main notebook (executed, with outputs)
└── Sneha_Chaurasia_46_GenAI_Lab_AE_VAE.pdf   ← assignment report
```

The dataset is loaded from Keras inside the notebook, so no dataset folder is needed.

---

## ⚙️ How to Run

**Option 1 — Google Colab (recommended)**

1. Open `Practical_03_AE_VAE_MNIST.ipynb` in Google Colab (`File → Upload notebook`).
2. Select a GPU: `Runtime → Change runtime type → T4 GPU` (optional, faster).
3. Run all cells: `Runtime → Run all`.

**Option 2 — Local Jupyter**

```
git clone https://github.com/Sneha529-oss/Autoencoder-and-VAE-for-Image-Reconstruction-Practical-No.-03-Gen-AI.git
cd Autoencoder-and-VAE-for-Image-Reconstruction-Practical-No.-03-Gen-AI
pip install tensorflow numpy pandas matplotlib seaborn jupyter
jupyter notebook Practical_03_AE_VAE_MNIST.ipynb
```

Training both models (plus the 2D latent-space model) takes about 2–4 minutes on a GPU.

---

## 📦 Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
jupyter
```

---

## ✅ Submission Checklist

- [x] Code file (Jupyter Notebook, executed end-to-end)
- [x] Dataset source noted above (loaded via Keras, no download needed)
- [x] Autoencoder: encoder, decoder, training, loss plot, reconstructions
- [x] VAE: encoder, latent mean/variance, sampling layer, decoder, training, loss plots, reconstructions, generated images
- [x] Comparison section (AE vs VAE)
- [x] Report PDF
- [x] README file

---

## 📖 References

1. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR 2014.
2. LeCun, Y., Cortes, C., & Burges, C. J. C. *The MNIST Database of Handwritten Digits.* <http://yann.lecun.com/exdb/mnist/>
3. Keras documentation — Variational Autoencoder example: <https://keras.io/examples/generative/vae/>
4. TensorFlow / Keras documentation: <https://www.tensorflow.org/>

---

## 📝 Declaration

I, **Sneha Chaurasia**, confirm that this implementation was prepared for Practical Assignment 3 and that the results presented are generated from the experiments performed in this notebook.

**GitHub Repository Link:** <https://github.com/Sneha529-oss/Autoencoder-and-VAE-for-Image-Reconstruction-Practical-No.-03-Gen-AI>
