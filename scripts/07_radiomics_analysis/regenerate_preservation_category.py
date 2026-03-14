#!/usr/bin/env python3
"""Regenerate preservation by category figure from results JSON."""

import json
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'radiomics_figures',
    Path(__file__).parent / 'radiomics_figures.py'
)
radiomics_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(radiomics_figures)

results_path = Path(__file__).parent.parent.parent / 'experiments' / 'radiomics_analysis' / 'radiomics_preservation_results.json'
with open(results_path) as f:
    results = json.load(f)

domain_a = results['domain_a_preservation']
output_path = Path(__file__).parent.parent.parent / 'figures' / 'radiomics' / 'fig_radiomics_preservation_category.pdf'
radiomics_figures.plot_preservation_by_category(domain_a, output_path)
print(f'Done: {output_path}')
