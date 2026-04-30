#!/bin/bash
# regenerate every figure, table, statistical test, and latex artifact for the
# journal extension. assumes:
#   - per-lambda training_history.json files exist in results/patchnce/
#   - patchnce_testset_summary.json exists in results/patchnce/
#   - all_results.json exists in results/
#
# usage: bash journal_extension/scripts/regenerate_all_artifacts.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$ROOT_DIR"

echo "========================================="
echo " regenerating extension a figures..."
echo "========================================="
python3 "$SCRIPT_DIR/generate_patchnce_figures.py"

echo "========================================="
echo " regenerating extension b/c/d/e figures..."
echo "========================================="
python3 "$SCRIPT_DIR/generate_figures.py"

echo "========================================="
echo " regenerating cross-extension synthesis..."
echo "========================================="
python3 "$SCRIPT_DIR/generate_cross_extension_figures.py"

echo "========================================="
echo " regenerating per-slice robustness analyses..."
echo "========================================="
python3 "$SCRIPT_DIR/generate_robustness_figures.py"

echo "========================================="
echo " regenerating statistical tests..."
echo "========================================="
python3 "$SCRIPT_DIR/statistical_tests.py"

FIG_DIR="$ROOT_DIR/journal_extension/figures"
RES_DIR="$ROOT_DIR/journal_extension/results"
MAN_DIR="$ROOT_DIR/journal_extension/manuscript"

echo "========================================="
echo " summary"
echo "========================================="
echo "  figures (.pdf): $(find "$FIG_DIR" -maxdepth 1 -type f -name '*.pdf' | wc -l)"
echo "  figures (.png): $(find "$FIG_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "  latex tables  : $(find "$FIG_DIR" -maxdepth 1 -type f -name '*.tex' | wc -l)"
echo "  result jsons  : $(find "$RES_DIR" -type f -name '*.json' | wc -l)"
echo "  manuscript    : $(find "$MAN_DIR" -type f -name '*.tex' | wc -l) tex, "\
"$(find "$MAN_DIR" -type f -name '*.bib' | wc -l) bib"
echo "  done."
