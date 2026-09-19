# LEMMA paper

This directory contains the source and published PDF for **LEMMA: Learned Guidance for Evidence-Carrying Long-Horizon Symbolic Rewriting**.

## Paper record

Saxena, Atul, and Pushp Kharat. 2026. *LEMMA: Learned Guidance for Evidence-Carrying Long-Horizon Symbolic Rewriting* (Version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22842820

The archival PDF is hosted on [Zenodo](https://doi.org/10.5281/zenodo.22842820). The benchmark and evaluated checkpoint are available on [Hugging Face](https://huggingface.co/datasets/BlackdromeAILabs/lemma-long-horizon-rewrite-benchmark) and [Hugging Face Models](https://huggingface.co/BlackdromeAILabs/lemma-v3-policy), respectively.

## Build

Compile `main.tex` with a LaTeX distribution that includes TikZ and `latexmk`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`LEMMA_PAPER.pdf` is the released PDF.
