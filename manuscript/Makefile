# Build the SPECTRA manuscript PDF.
#   make          build spectra.pdf (latexmk if installed, otherwise tectonic)
#   make clean    remove build files
# Figures are read from ../figures/main and ../figures/supplementary (public repository layout)
# or ../figures/manuscript_20261111/final_figures (project layout).

MAIN = spectra

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic --keep-logs $(MAIN).tex; \
	else \
		echo "No LaTeX engine found: install TeX Live (latexmk) or Tectonic (https://tectonic-typesetting.github.io)."; \
		exit 1; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -C $(MAIN).tex; fi
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).blg $(MAIN).bbl

.PHONY: all clean
