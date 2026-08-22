# A Short Course for Modern Large Language Models

An 18-lesson course that builds the mathematical and implementation intuition needed to understand a modern decoder-only Transformer language model.

**Live course:** [jerryyangboyu.github.io](https://jerryyangboyu.github.io/)

## Course Goal

By the end of the course, a student should be able to:

- derive the linear algebra and trigonometry behind rotary position encoding;
- track tensor shapes through normalization, attention, residual paths, and SwiGLU;
- explain how causal self-attention converts query-key comparisons into weighted value mixtures;
- trace a decoder block from token embeddings to next-token logits; and
- read, run, and modify the included PyTorch `TransformerModel`.

RoPE is taught as one positional-encoding component within the larger Transformer architecture, not as the course's final subject.

## Curriculum

### Module 1: Mathematical Foundations

Lessons 1-9 develop the prerequisites used by positional encoding and attention:

- vectors and paired dimensions;
- radians, the unit circle, and trigonometry;
- complex numbers and rotations;
- matrices, dot products, and transposes;
- attention as vector comparison; and
- position-dependent frequencies and relative position.

### Module 2: One Transformer Decoder Block

Lessons 10-18 follow the implementation path of a decoder-only Transformer:

- hidden states and tensor shapes;
- LayerNorm, RMSNorm, and pre-norm placement;
- Q, K, and V projections;
- rotary position encoding;
- causal masking, softmax, and weighted values;
- multi-head recombination;
- residual connections;
- SwiGLU feed-forward layers; and
- the complete `TransformerModel`.

Each Module 2 lesson includes a focused, syntax-highlighted PyTorch code checkpoint. The checkpoints accumulate into the final runnable implementation.

## Reference Implementation

- [Annotated implementation](https://jerryyangboyu.github.io/reference/transformer-model-code.html)
- [Complete Python source](reference/transformer_model.py)
- [Transformer block visual map](https://jerryyangboyu.github.io/reference/transformer-block-visual.html)
- [Transformer block formula sheet](https://jerryyangboyu.github.io/reference/transformer-block-reference.html)

The teaching model is adapted from MiniMind's `model/model_minimind.py`. It retains the core decoder path while omitting production features such as grouped-query attention, KV caching, Flash Attention, YaRN scaling, mixture-of-experts routing, and Hugging Face generation wrappers.

Run the reference model with:

```bash
python reference/transformer_model.py
```

Expected output shapes:

```text
input: (1, 4)
logits: (1, 4, 32)
```

## Study Method

For each lesson:

1. Read the explanation and predict the result before revealing it.
2. Work through the numerical examples by hand.
3. Complete the exercises before opening the solutions.
4. Trace the lesson's code checkpoint and annotate every tensor shape.
5. Explain the concept aloud without relying on the page.

The full course takes approximately four hours and assumes no calculus.

## Local Preview

The site is plain HTML, CSS, and JavaScript. From the repository root, run:

```bash
python -m http.server 8765
```

Then open [http://localhost:8765/](http://localhost:8765/).

## Repository Layout

```text
assets/       Course styles and interactive visualizations
lessons/      Course index and 18 lessons
reference/    Formula sheets, visual maps, and TransformerModel source
index.html    GitHub Pages entry point
```
