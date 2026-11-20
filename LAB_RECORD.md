# Lab Record: ETMMGA304 Generative AI and Its Applications

**Student:** ____________________  **Roll no.:** ____________  **Programme:** MCA

Fill in each blank table from **your own run**. The *expected ranges* below
are approximate values for the default settings, to help you spot a broken
run. They are not results to copy. Values change with hardware, library
versions and random seeds.

Write down the following for every experiment:

| Field | Value |
|-------|-------|
| Date | |
| Device (CPU / GPU model) | |
| Python / PyTorch / Transformers versions | |
| Seed | 42 |

---

## Lab 1: Variational Autoencoder for Image Generation

**Aim.** Build and train a VAE in PyTorch on MNIST. Generate handwritten
digits, compare reconstructions with the original images, and visualise the
2-D latent space.

**Software / data.** PyTorch, torchvision MNIST (60k train / 10k test),
Matplotlib. Script: `lab1_vae_mnist.py`.

**Parameters.** Encoder 784→256→(μ, log σ²) ∈ ℝ²; decoder 2→256→784 with a
Sigmoid output. Adam, lr = 0.001, batch size 128, 5 epochs, seed 42.

### Observation table

| Epoch | Reconstruction loss / image | KL divergence / image | Comments |
|-------|-----------------------------|-----------------------|----------|
| 1 | | | |
| 3 | | | |
| 5 | | | |

**Expected ranges.** Reconstruction BCE is about 175–200 per image at epoch 1
and falls to about 145–160 by epoch 5. KL is about 4–6 per image and usually
*rises* slightly over training (to about 6–7), because the encoder spreads
the codes out to carry more information. Both values must stay finite.

| Visual check | Your value |
|--------------|------------|
| Recognisable digits out of 16 (`vae_generated.png`) | / 16 |
| Two labels whose latent points overlap (`vae_latent.png`) | |
| One region with a clear grouping | |

Typical outcome: 8–13 of 16 samples are readable. 4/9 and 3/5/8 overlap in
the latent plot, while 0 and 1 form well-separated clusters.

### Result

Reconstructions keep the digit identity and overall shape but are blurrier
than the originals. Samples drawn from the prior give plausible digits
wherever the prior overlaps the learned clusters.

**Limitation.** A 2-D latent space is too small to hold fine stroke detail,
and samples from regions between clusters look like blends of two digits.
**Next step.** Increase the latent size to 16–32 (and plot it with t-SNE or
PCA), or use a convolutional encoder and decoder.

### Viva answers

1. **What do the mean and log variance represent?**
   They are the parameters of the approximate posterior q(z|x) = 𝒩(μ, σ²).
   μ is the centre of the region of latent space the encoder assigns to the
   image. log σ² is how uncertain that position is. The encoder predicts the
   log variance instead of σ² because log σ² can be any real number, while
   σ² must be positive. This keeps training numerically stable.

2. **Why is the KL term included in the objective?**
   KL(q(z|x) ‖ 𝒩(0, I)) pulls every encoding towards the standard-normal
   prior. Without it the encoder could give each image a tiny, isolated
   code, and the model would become a plain autoencoder. Decoding random
   z ~ 𝒩(0, I) would then land in empty space and give garbage. The KL term
   makes the latent space continuous and densely filled, which is what makes
   sampling work. Together with the reconstruction term, it forms the
   negative ELBO.

3. **How does a reconstructed digit differ from a newly generated digit?**
   A reconstruction decodes μ(x) for a real test image, so it is tied to
   that input and should look like it. A generated digit decodes a fresh
   z ~ 𝒩(0, I) with no input image, so it shows what the model has learned
   about the whole data distribution. Generated digits are usually less
   sharp and sometimes ambiguous.

4. **Why use held-out images to draw the latent scatterplot?**
   Test images show how the encoder handles data it never trained on, so
   the plot shows whether the latent structure generalises. Training images
   could show clusters that only reflect memorisation.

---

## Lab 2: Image Synthesis Using GANs

**Aim.** Train a basic GAN on Fashion-MNIST. Watch how outputs for fixed
noise inputs change across epochs, and judge training from both the images
and the losses.

**Software / data.** PyTorch, torchvision Fashion-MNIST (60k train), pixels
normalised to [−1, 1]. Script: `lab2_gan_fashion_mnist.py`.

**Parameters.** G: 64→256→512→784 (LeakyReLU 0.2, Tanh output). D:
784→512→256→1 logit. BCEWithLogitsLoss. Adam, lr = 2e-4, β = (0.5, 0.999),
batch size 128, 10 epochs, seed 42, 25 fixed noise vectors.

### Observation table

| Epoch | D loss | G loss | Recognisable out of 25 | Repeated shapes or artefacts |
|-------|--------|--------|------------------------|------------------------------|
| 1 | | | | |
| 5 | | | | |
| 10 | | | | |

**Expected ranges.** D loss is the mean of its real and fake BCE, so it
usually settles between about 0.45 and 0.65. G loss is about 0.9–2.0 and is
often highest in the first epoch. Neither loss should go to 0 or keep
climbing. At epoch 1 the samples are noisy blobs with rough silhouettes
(0–8 of 25 recognisable). By epoch 10, trousers, shirts, sneakers and bags
are clearly visible (15–22 of 25), with some speckle noise and soft edges.

### Result

Over training, the same grid positions sharpen from blobs into clear
garment silhouettes. Flat losses that do not diverge, together with varied
classes across the grid, show that training is stable. GAN losses do not
fall steadily the way supervised losses do, because each network's loss
depends on the other network's progress.

**Limitation.** MLP networks have no spatial inductive bias, so textures
stay noisy. Partial mode collapse can appear as one silhouette repeated in
many grid positions. **Next step.** Use a DCGAN (convolutional) architecture,
train for longer, and measure FID instead of judging by eye.

### Viva answers

1. **What does each network learn during training?**
   The discriminator learns a classifier that outputs a high logit for real
   Fashion-MNIST images and a low logit for generated ones. The generator
   learns to map noise z to images that the discriminator scores as real,
   which pushes its output distribution towards the data distribution. The
   two networks play a minimax game.

2. **Why is the generated batch detached in the discriminator step?**
   `fake.detach()` cuts the computation graph, so `d_loss.backward()`
   computes gradients for D only. Without the detach, gradients would also
   flow into G, wasting compute, and could move G in the direction that helps
   D (the opposite of what G should learn). The generator step then reuses
   the non-detached `fake`, with D frozen.

3. **Why are the grid inputs kept fixed?**
   With `fixed_z` reused at every checkpoint, each grid cell shows the same
   latent point at a different stage of training. Differences between grids
   then come only from changes to the generator, not from new random noise,
   so progress and collapse are easy to compare.

4. **How can a grid reveal mode collapse?**
   With 25 different noise vectors, a healthy generator produces many
   garment classes. If many cells show nearly the same silhouette, or the
   variety shrinks from epoch 5 to epoch 10, the generator has collapsed onto
   a few modes that fool D. The losses alone can look normal while this
   happens.

---

## Lab 3: Text Generation Using GPT-2

**Aim.** Use GPT-2 to generate continuations, compare decoding settings, and
fine-tune the last Transformer block to write short study tips.

**Software / model.** Hugging Face Transformers, `openai-community/gpt2`
(124M parameters). Script: `lab3_gpt2_text_generation.py` (outputs are also
saved to `lab_outputs/lab3_decoding.csv`).

**Parameters.** `max_new_tokens = 45`. Sampling uses top_p = 0.9, T ∈ {0.6,
1.0} and seeds {42, 43}. Fine-tuning: only `transformer.h[-1]` is trainable
(about 7.1M parameters), AdamW, lr = 2e-5, 3 epochs over 6 examples,
prompt tokens masked with −100.

### Decoding observation table (0 = poor, 1 = partial, 2 = strong)

| Prompt | Decoding and seed | Coherence 0–2 | Relevance 0–2 | Creativity 0–2 | Supporting excerpt |
|--------|-------------------|---------------|---------------|----------------|--------------------|
| Robot story | Greedy | | | | |
| Robot story | T 0.6, seed 42 | | | | |
| Robot story | T 0.6, seed 43 | | | | |
| Robot story | T 1.0, seed 42 | | | | |
| Robot story | T 1.0, seed 43 | | | | |
| Library story | Greedy | | | | |
| Library story | T 0.6, seed 42 | | | | |
| Library story | T 0.6, seed 43 | | | | |
| Library story | T 1.0, seed 42 | | | | |
| Library story | T 1.0, seed 43 | | | | |

**Typical pattern.** Greedy output is fluent but tends to repeat a phrase or
loop (high coherence, low creativity). T = 0.6 stays on topic with some
variety. T = 1.0 is the most novel but drifts off topic or loses grammar
more often. The more specific library prompt keeps all methods more
relevant than the open-ended robot prompt.

### Held-out comparison (relevance 0–2 + usefulness 0–2)

| Held-out topic | Base score / 4 | Tuned score / 4 | Example difference |
|----------------|----------------|-----------------|--------------------|
| Recursion | | | |
| Matrices | | | |
| Binary search | | | |

**Expected behaviour.** The last-example loss falls only a little over 3
epochs (for example from about 4–5 to about 3–4). With six examples and a
small learning rate, the tuned model may pick up the *format* (one short
imperative sentence ending at EOS) more than the *content*. Base GPT-2
often rambles, or continues the "Topic:" pattern with new topics. Any
improvement is small and inconsistent. As the manual notes, a six-example
fine-tune shows the workflow; it is not evidence of reliable improvement.

### Result and limitation

Decoding strategy controls a trade-off between safe and repetitive output
(greedy, low T) and varied but risky output (high T). **Limitation:** the
dataset is tiny, so the tuned outputs can be generic or factually empty.
**Next step:** use 100+ varied, approved pairs, a validation split, and more
epochs.

### Viva answers

1. **How does greedy decoding select the next token?**
   At each step it takes the arg-max of the model's next-token
   distribution, appends that token, and repeats. It is deterministic, so
   seeds have no effect. It often falls into repetitive loops, because the
   locally most likely token is not always part of the best sequence.

2. **How does temperature affect sampling?**
   The logits are divided by T before the softmax. T < 1 sharpens the
   distribution towards the most likely tokens (more conservative). T > 1
   flattens it so less likely tokens are picked more often (more diverse,
   less coherent). Here top-p = 0.9 is also applied, which samples only
   from the smallest set of tokens whose total probability reaches 0.9.

3. **Which tokens contribute to the fine-tuning loss?**
   Only the answer tokens: the study tip and the final EOS. Prompt tokens
   ("Topic: …\nStudy tip:") get the label −100, which the cross-entropy
   loss ignores. So the model learns to *answer* the prompt, not to
   reproduce it, and it learns to stop at EOS.

4. **Why are the held-out topics excluded from training?**
   To measure generalisation. If recursion, matrices or binary search were
   in the training set, a good score could just mean the model memorised
   the answer. Unseen topics test whether the model learned the task.

---

## Lab 4: Multimodal Prompt-to-Image Generation

**Aim.** Generate images from text with Stable Diffusion, changing one prompt
element (the style) while keeping the seed and settings the same.

**Software / model.** Diffusers, `stable-diffusion-v1-5/stable-diffusion-v1-5`,
float16 on a CUDA GPU with attention slicing. Script:
`lab4_stable_diffusion.py`.

**Parameters.** 512×512, 25 inference steps, guidance scale 7.5, seeds 42
and 43, built-in safety checker enabled.

### Observation table (0 = missing, 1 = partial, 2 = clear)

| Style requested | Seed | Subject score 0–2 | Style score 0–2 | Composition or errors |
|-----------------|------|-------------------|-----------------|-----------------------|
| Watercolor | 42 | | | |
| Watercolor | 43 | | | |
| Pencil sketch | 42 | | | |
| Pencil sketch | 43 | | | |

**Typical pattern.** At the same seed, the two styles often share a similar
layout (camera angle, position of the table and shelves), because the
starting noise sets the coarse structure. Changing the style mainly changes
colour, texture and line quality (soft colour washes compared with
monochrome hatching). Changing the seed changes layout, viewpoint and
object placement. Common errors include warped shelves and book spines,
gibberish text on books, and chairs with impossible geometry.

If you use Replicate or the DALL·E API instead, record the model version,
prompts, seed and settings, and only compare images made with the same
backend.

### Result, application and risk

**Application:** quick concept art and illustrations for teaching material
or storyboards. **Risk:** models trained on web-scraped images can copy
artists' styles without permission and repeat demographic bias (for
example, who appears in a "librarian" image). Photorealistic outputs can
also be passed off as real photographs.

### Viva answers

1. **Why should prompt variants be compared at the same seed?**
   The seed fixes the initial latent noise, which largely sets the
   composition. With the seed and all other settings held constant, the
   prompt is the only thing that changes, so any difference can be credited
   to the wording. Changing both at once would mix the two effects.

2. **What is the purpose of denoising steps?**
   Diffusion models learn to reverse a gradual noising process. At each
   step the U-Net predicts the noise in the current latent, guided by the
   text embedding, and the scheduler removes part of it. Over 25 steps, pure
   noise is refined into a clean latent, which the VAE decoder turns into
   pixels. More steps give finer detail, up to a point, at the cost of speed.

3. **How can you rate subject and style adherence independently?**
   Use separate rubric questions. *Subject:* is there a small library with
   a reading table in a wide view, whatever the medium? *Style:* does the
   rendering look like watercolour or pencil, whatever the content? An image
   can score 2 on subject and 0 on style (a photo-real library), or the
   reverse. Scoring each one blind to the other, or by two raters, reduces
   halo bias.

4. **When should a generated image carry a disclosure?**
   Whenever viewers could mistake it for a real photograph or human-made
   work, or when it is used in news, education, advertising, academic
   submissions, or anything showing real people or places. Disclosure is
   also required wherever the platform, institution or law demands it. A
   visible label, plus provenance metadata such as C2PA, supports
   transparency and helps prevent misinformation.

---

## Capstone: Generative AI Creative Solution (Study Tip Generator)

**Define.** Beginners often do not know *how* to start studying a new
topic. The app takes a topic and returns one short, actionable study tip.
**Success criteria:** mean adapted score ≥ 4/6 on at least 10 held-out
topics, and higher than the base model.

**Prepare.** Instructor-approved (topic, tip) pairs, with separate training
and held-out topic lists that do not overlap. Record the source,
permissions and split sizes.

**Adapt.** GPT-2 with only the last block trained (Lab 3 workflow).
Settings: lr 2e-5, 3 or more epochs, AdamW. Keep the loss log and the
`lab_outputs/study_tip_gpt2` checkpoint.

**Demonstrate.** `streamlit run capstone_app.py`

### Evaluation record (≥ 10 held-out topics; each criterion 0–2)

| Topic | Model | Relevance | Readability | Usefulness | Total / 6 | Notes |
|-------|-------|-----------|-------------|------------|-----------|-------|
| recursion | Base | | | | | |
| recursion | Adapted | | | | | |
| matrices | Base | | | | | |
| matrices | Adapted | | | | | |
| binary search | Base | | | | | |
| binary search | Adapted | | | | | |
| linked lists | Base | | | | | |
| linked lists | Adapted | | | | | |
| derivatives | Base | | | | | |
| derivatives | Adapted | | | | | |
| SQL joins | Base | | | | | |
| SQL joins | Adapted | | | | | |
| hash tables | Base | | | | | |
| hash tables | Adapted | | | | | |
| logarithms | Base | | | | | |
| logarithms | Adapted | | | | | |
| stacks and queues | Base | | | | | |
| stacks and queues | Adapted | | | | | |
| Big-O notation | Base | | | | | |
| Big-O notation | Adapted | | | | | |
| **Mean** | **Base** | | | | | |
| **Mean** | **Adapted** | | | | | |

**Failure analysis (at least one).** Example topic: ______. Output:
"______". Why it fails: ______ (for example, a generic tip that would fit
any topic, a factually wrong statement, or text that continues with a new
"Topic:" line).

**Conditions for use.** For teacher-reviewed drafting only. Not for
unsupervised student use. English topics only. Short factual topics only.

### Demonstration checks

| Check | Pass? | Notes |
|-------|-------|-------|
| Empty topic shows the "Enter a study topic first." warning | | |
| Missing model folder shows an error and stops | | |
| Held-out topic 1 gives an understandable, useful tip | | |
| Held-out topic 2 gives an understandable, useful tip | | |
| One weak output presented, with how a reviewer would handle it | | |

### Privacy, bias and limitations (disclosure)

- **Disclosure:** the app labels every tip as AI-generated and meant for
  instructor review.
- **Privacy:** the model runs locally. Topics are not logged or sent to a
  third party. Do not enter personal data.
- **Bias:** GPT-2 was trained on WebText (Reddit-linked pages), which
  leans towards English, Western and male-authored content. Tips may assume
  particular resources or learning styles.
- **Limitations:** a 124M-parameter model can state false facts with
  confidence. Output length is fixed at 45 tokens. Greedy decoding always
  gives the same answer for a topic.

### Viva answers

1. **Why did you select this dataset and model?**
   GPT-2 is open, small enough to fine-tune on CPU or a free Colab GPU, and
   already writes fluent English, so only task formatting needs to be
   learned. Training just the last block cuts compute and the risk of
   over-fitting a small dataset. Instructor-approved (topic, tip) pairs give
   the model data that is accurate, appropriate and properly licensed.

2. **Which results show whether adaptation improved the outputs?**
   The paired comparison on the same held-out topics: the mean total score
   out of 6 for base and adapted models, per-criterion differences, and raw
   example outputs. Training loss alone is not evidence, because it is
   measured on seen examples. A higher held-out mean, with consistent gains
   across most topics, shows real improvement.

3. **What human review would you require before students use the app?**
   An instructor checks each generated tip for accuracy, clarity and
   suitability before it reaches students, using an approved list or a
   moderation queue. Spot-checks are repeated whenever the model or data
   changes, and students get a way to report bad tips. The interface states
   clearly that the content is AI-generated.
