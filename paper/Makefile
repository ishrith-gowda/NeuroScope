# ==============================================================================
# Makefile for SA-CycleGAN Paper
# ==============================================================================

# Main document
MAIN = main
SUPP = supplementary
REBUTTAL = rebuttal

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Output directory
OUTDIR = build

# Default target
all: paper supplementary

# Create output directory
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Build main paper
paper: $(OUTDIR)
	@echo "Building main paper..."
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(BIBTEX) $(OUTDIR)/$(MAIN)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .
	@echo "Main paper built: $(MAIN).pdf"

# Build supplementary materials
supplementary: $(OUTDIR)
	@echo "Building supplementary materials..."
	$(LATEX) -output-directory=$(OUTDIR) $(SUPP).tex
	$(LATEX) -output-directory=$(OUTDIR) $(SUPP).tex
	cp $(OUTDIR)/$(SUPP).pdf .
	@echo "Supplementary built: $(SUPP).pdf"

# Build rebuttal
rebuttal: $(OUTDIR)
	@echo "Building rebuttal..."
	$(LATEX) -output-directory=$(OUTDIR) $(REBUTTAL).tex
	$(LATEX) -output-directory=$(OUTDIR) $(REBUTTAL).tex
	cp $(OUTDIR)/$(REBUTTAL).pdf .
	@echo "Rebuttal built: $(REBUTTAL).pdf"

# Quick build (single pass)
quick: $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .

# Clean auxiliary files
clean:
	rm -rf $(OUTDIR)
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk
	rm -f *.bbl *.blg *.synctex.gz

# Clean all including PDFs
cleanall: clean
	rm -f $(MAIN).pdf $(SUPP).pdf $(REBUTTAL).pdf

# Watch for changes and rebuild
watch:
	@echo "Watching for changes..."
	@while true; do \
		inotifywait -q -e modify *.tex *.bib; \
		make paper; \
	done

# Word count
count:
	@echo "Word count (approximate):"
	@detex $(MAIN).tex | wc -w

# Check for common LaTeX errors
check:
	@echo "Checking for common issues..."
	@grep -n "TODO\|FIXME\|XXX" $(MAIN).tex $(SUPP).tex || true
	@grep -n "\\\\cite{}" $(MAIN).tex || true
	@grep -n "\\\\ref{}" $(MAIN).tex || true
	@echo "Check complete."

# Archive for submission
archive: paper supplementary
	@echo "Creating submission archive..."
	mkdir -p submission
	cp $(MAIN).pdf submission/
	cp $(SUPP).pdf submission/
	cp $(MAIN).tex submission/
	cp $(SUPP).tex submission/
	cp references.bib submission/
	cp neurips_2024.sty submission/
	cp -r ../figures/generated submission/figures
	cd submission && zip -r ../submission.zip .
	rm -rf submission
	@echo "Archive created: submission.zip"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build main paper and supplementary"
	@echo "  paper        - Build main paper only"
	@echo "  supplementary - Build supplementary materials"
	@echo "  rebuttal     - Build rebuttal document"
	@echo "  quick        - Quick build (single pass)"
	@echo "  clean        - Remove auxiliary files"
	@echo "  cleanall     - Remove all generated files"
	@echo "  watch        - Watch for changes and rebuild"
	@echo "  count        - Word count"
	@echo "  check        - Check for common issues"
	@echo "  archive      - Create submission archive"
	@echo "  help         - Show this help"

.PHONY: all paper supplementary rebuttal quick clean cleanall watch count check archive help
