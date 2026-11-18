# Manuscript

`spectra.pdf` is the compiled manuscript; `spectra.tex` is its LaTeX source of the SPECTRA manuscript, with the main figures, tables, references and Supplementary Information (Supplementary Figs. 1–6 and Supplementary Table 1).

Build the PDF with:

```bash
make
```

This uses `latexmk` (TeX Live) if it is installed, and otherwise [Tectonic](https://tectonic-typesetting.github.io). Figures are read from `../figures/main` and `../figures/supplementary`.

Line numbers are on for review; comment out `\linenumbers` in the preamble for a clean reading copy.
