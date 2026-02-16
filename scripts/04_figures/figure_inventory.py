"""
figure inventory for neuroscope publication

this module contains the complete catalog of all publication figures
with their descriptions, generation scripts, and data dependencies.

all figures are stored in /figures/ directory organized by category.
only pdf versions are kept; svg/png can be regenerated when needed.

total figures: 62 pdfs + 9 latex tables
"""

# figure inventory organized by category
# format: {filename: {description, script, data_dependency, notes}}

TRAINING_FIGURES = {
    "fig01_training_losses.pdf": {
        "description": "Training Loss Curves (Generator, Discriminator, Cycle, Identity)",
        "script": "generate_training_figures.py",
        "data": "results/training/training_history.json",
        "notes": "shows convergence over 100 epochs"
    },
    "fig02_validation_metrics.pdf": {
        "description": "Validation Metrics Over Training (SSIM, PSNR for A2B and B2A)",
        "script": "generate_training_figures.py",
        "data": "results/training/training_history.json",
        "notes": "demonstrates model improvement during training"
    },
    "fig04_learning_rate.pdf": {
        "description": "Learning Rate Schedule (Cosine with Warmup)",
        "script": "generate_training_figures.py",
        "data": "results/training/training_history.json",
        "notes": "shows cosine annealing with warmup"
    },
    "fig08_dataset_statistics.pdf": {
        "description": "Dataset Statistics (BraTS + UPenn volume distributions)",
        "script": "generate_dataset_figures.py",
        "data": "data/metadata.json",
        "notes": "shows dataset composition and modality distribution"
    },
    "fig09_preprocessing_pipeline.pdf": {
        "description": "Preprocessing Pipeline Flowchart",
        "script": "generate_dataset_figures.py",
        "data": None,
        "notes": "schematic diagram of preprocessing steps"
    },
    "fig10_25d_processing.pdf": {
        "description": "2.5D Processing Concept Visualization",
        "script": "generate_dataset_figures.py",
        "data": None,
        "notes": "explains 2.5D slice extraction strategy"
    },
    "fig11_training_overview.pdf": {
        "description": "Training Overview Schematic",
        "script": "generate_dataset_figures.py",
        "data": None,
        "notes": "high-level training architecture diagram"
    }
}

ABLATION_FIGURES = {
    "fig_ablation_study.pdf": {
        "description": "Ablation Study: Baseline vs Attention Model Comparison",
        "script": "generate_ablation_figure.py",
        "data": "evaluation_results/ablation/ablation_results_*.json",
        "notes": "bar charts comparing baseline cyclegan vs sa-cyclegan metrics"
    }
}

DOWNSTREAM_FIGURES = {
    "fig_domain_classification.pdf": {
        "description": "Domain Classification Results (Pre/Post Harmonization)",
        "script": "05_downstream_evaluation/generate_downstream_figures.py",
        "data": "experiments/downstream_evaluation/domain_classification_results.json",
        "notes": "shows reduced domain discriminability after harmonization"
    },
    "fig_feature_distribution.pdf": {
        "description": "Feature Distribution Analysis (t-SNE)",
        "script": "05_downstream_evaluation/generate_downstream_figures.py",
        "data": "experiments/downstream_evaluation/feature_distribution_results.json",
        "notes": "visualizes feature space alignment"
    },
    "fig_harmonization_summary.pdf": {
        "description": "Harmonization Summary with All Metrics",
        "script": "05_downstream_evaluation/generate_downstream_figures.py",
        "data": "experiments/downstream_evaluation/evaluation_summary.json",
        "notes": "comprehensive summary of harmonization effectiveness"
    },
    "fig_training_curves.pdf": {
        "description": "Domain Classifier Training Curves",
        "script": "05_downstream_evaluation/generate_downstream_figures.py",
        "data": "experiments/downstream_evaluation/training_history.json",
        "notes": "shows domain classifier training convergence"
    },
    "fig_tsne_visualization.pdf": {
        "description": "t-SNE Visualization of Domain Shift",
        "script": "05_downstream_evaluation/generate_downstream_figures.py",
        "data": "experiments/downstream_evaluation/tsne_*.npy",
        "notes": "2d projection showing domain alignment"
    }
}

STATISTICAL_FIGURES = {
    "fig06_metric_distributions.pdf": {
        "description": "Distribution of Image Quality Metrics",
        "script": "generate_quantitative_figures.py",
        "data": "results/evaluation/evaluation_results.json",
        "notes": "histograms of ssim, psnr, lpips distributions"
    },
    "fig07_cycle_consistency.pdf": {
        "description": "Cycle Consistency Analysis",
        "script": "generate_quantitative_figures.py",
        "data": "results/evaluation/cycle_consistency_results.json",
        "notes": "measures reconstruction quality"
    },
    "fig15_comprehensive_comparison.pdf": {
        "description": "Comprehensive Method Comparison",
        "script": "generate_statistical_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "compares all methods across metrics"
    },
    "fig16_approximated_distributions.pdf": {
        "description": "Approximated Metric Distributions",
        "script": "generate_statistical_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "shows distribution approximations for statistics"
    },
    "fig17_performance_radar.pdf": {
        "description": "Performance Radar Chart (Multi-Metric)",
        "script": "generate_statistical_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "spider/radar plot of all performance metrics"
    },
    "fig19_effect_size_analysis.pdf": {
        "description": "Effect Size Analysis (Cohen's d)",
        "script": "generate_statistical_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "statistical effect sizes for method comparisons"
    },
    "fig_improvement_waterfall.pdf": {
        "description": "Improvement Waterfall Chart",
        "script": "06_statistical_analysis/run_statistical_analysis.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "shows cumulative improvement from each component"
    },
    "fig_method_comparison.pdf": {
        "description": "Method Comparison Bar Charts",
        "script": "06_statistical_analysis/run_statistical_analysis.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "compares sa-cyclegan vs combat vs baseline"
    },
    "fig_radar_comparison.pdf": {
        "description": "Multi-Method Radar Comparison",
        "script": "06_statistical_analysis/run_statistical_analysis.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "radar chart comparing all harmonization methods"
    },
    "fig_statistical_summary.pdf": {
        "description": "Statistical Summary Panel",
        "script": "06_statistical_analysis/run_statistical_analysis.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "comprehensive statistical test results"
    }
}

RADIOMICS_FIGURES = {
    "fig_radiomics_bland_altman.pdf": {
        "description": "Bland-Altman Plots for Radiomic Features",
        "script": "07_radiomics_analysis/radiomics_figures.py",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json",
        "notes": "agreement analysis for radiomic features"
    },
    "fig_radiomics_comprehensive.pdf": {
        "description": "Comprehensive Radiomics Preservation Analysis",
        "script": "07_radiomics_analysis/radiomics_figures.py",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json",
        "notes": "full panel of radiomics preservation metrics"
    },
    "fig_radiomics_correlation_heatmap.pdf": {
        "description": "Radiomics Feature Correlation Heatmap",
        "script": "07_radiomics_analysis/radiomics_figures.py",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json",
        "notes": "correlation matrix of radiomic features pre/post"
    },
    "fig_radiomics_preservation_category.pdf": {
        "description": "Radiomics Preservation by Feature Category",
        "script": "07_radiomics_analysis/radiomics_figures.py",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json",
        "notes": "grouped analysis by radiomic feature type"
    },
    "fig_radiomics_scatter.pdf": {
        "description": "Radiomics Feature Scatter Plots",
        "script": "07_radiomics_analysis/radiomics_figures.py",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json",
        "notes": "pre vs post harmonization scatter"
    }
}

COMPUTATIONAL_FIGURES = {
    "fig_efficiency_comparison.pdf": {
        "description": "Computational Efficiency Comparison",
        "script": "09_computational_analysis/efficiency_analysis.py",
        "data": "experiments/computational_analysis/efficiency_results.json",
        "notes": "runtime, memory, throughput comparisons"
    }
}

PUBLICATION_FIGURES = {
    "fig01_training_curves.pdf": {
        "description": "Publication-Ready Training Curves",
        "script": "generate_publication_figures.py",
        "data": "results/training/training_history.json",
        "notes": "polished version for paper main body"
    },
    "fig02_ablation_comparison.pdf": {
        "description": "Publication-Ready Ablation Comparison",
        "script": "generate_publication_figures.py",
        "data": "evaluation_results/ablation/ablation_results_*.json",
        "notes": "polished ablation study results"
    },
    "fig03_modality_analysis.pdf": {
        "description": "Per-Modality Analysis (T1, T1CE, T2, FLAIR)",
        "script": "generate_publication_figures.py",
        "data": "results/evaluation/evaluation_results.json",
        "notes": "breakdown of results by mri modality"
    },
    "fig04_effect_sizes.pdf": {
        "description": "Publication-Ready Effect Size Analysis",
        "script": "generate_publication_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "polished statistical effect sizes"
    },
    "fig05_radar_comparison.pdf": {
        "description": "Publication-Ready Radar Chart",
        "script": "generate_publication_figures.py",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json",
        "notes": "polished multi-metric radar comparison"
    },
    "fig06_improvement_by_modality.pdf": {
        "description": "Improvement Analysis by Modality",
        "script": "generate_publication_figures.py",
        "data": "results/evaluation/evaluation_results.json",
        "notes": "shows improvement for each mri sequence"
    },
    "fig07_summary_table.pdf": {
        "description": "Summary Table as Figure",
        "script": "generate_publication_figures.py",
        "data": "multiple",
        "notes": "visual summary table for paper"
    },
    "fig08_direction_comparison.pdf": {
        "description": "Translation Direction Comparison (A2B vs B2A)",
        "script": "generate_publication_figures.py",
        "data": "results/evaluation/evaluation_results.json",
        "notes": "compares forward and backward translation"
    },
    "fig_comprehensive_results.pdf": {
        "description": "Comprehensive Results Summary Panel",
        "script": "10_publication_summary/comprehensive_results_figure.py",
        "data": "multiple",
        "notes": "main results figure for paper"
    }
}

ARCHITECTURE_FIGURES = {
    "fig12_architecture_comparison.pdf": {
        "description": "Architecture Comparison (Baseline vs SA-CycleGAN)",
        "script": "generate_architecture_figures.py",
        "data": None,
        "notes": "side-by-side architecture diagrams"
    },
    "fig13_attention_mechanisms.pdf": {
        "description": "Attention Mechanism Visualization",
        "script": "generate_architecture_figures.py",
        "data": None,
        "notes": "spatial and channel attention diagrams"
    },
    "fig14_parameter_breakdown.pdf": {
        "description": "Model Parameter Breakdown",
        "script": "generate_architecture_figures.py",
        "data": None,
        "notes": "pie chart of parameter distribution"
    },
    "fig15_cyclegan_workflow.pdf": {
        "description": "CycleGAN Workflow Diagram",
        "script": "generate_architecture_figures.py",
        "data": None,
        "notes": "full cyclegan training workflow"
    },
    "fig_architecture_25d_concept.pdf": {
        "description": "2.5D Architecture Concept",
        "script": "generate_architecture_diagram.py",
        "data": None,
        "notes": "explains 2.5d slice processing approach"
    },
    "fig_architecture_cyclegan.pdf": {
        "description": "CycleGAN Architecture Diagram",
        "script": "generate_architecture_diagram.py",
        "data": None,
        "notes": "full cyclegan architecture schematic"
    },
    "fig_architecture_generator.pdf": {
        "description": "Generator Network Architecture",
        "script": "generate_architecture_diagram.py",
        "data": None,
        "notes": "detailed generator network diagram"
    }
}

VISUAL_EXAMPLES = {
    "visual_sample_00_T1.pdf": {"description": "Sample 0 - T1 Translation", "modality": "T1"},
    "visual_sample_00_T1CE.pdf": {"description": "Sample 0 - T1CE Translation", "modality": "T1CE"},
    "visual_sample_00_T2.pdf": {"description": "Sample 0 - T2 Translation", "modality": "T2"},
    "visual_sample_00_FLAIR.pdf": {"description": "Sample 0 - FLAIR Translation", "modality": "FLAIR"},
    "visual_sample_00_all_modalities.pdf": {"description": "Sample 0 - All Modalities", "modality": "all"},
    "visual_sample_01_T1.pdf": {"description": "Sample 1 - T1 Translation", "modality": "T1"},
    "visual_sample_01_T1CE.pdf": {"description": "Sample 1 - T1CE Translation", "modality": "T1CE"},
    "visual_sample_01_T2.pdf": {"description": "Sample 1 - T2 Translation", "modality": "T2"},
    "visual_sample_01_FLAIR.pdf": {"description": "Sample 1 - FLAIR Translation", "modality": "FLAIR"},
    "visual_sample_01_all_modalities.pdf": {"description": "Sample 1 - All Modalities", "modality": "all"},
    "visual_sample_02_T1.pdf": {"description": "Sample 2 - T1 Translation", "modality": "T1"},
    "visual_sample_02_T1CE.pdf": {"description": "Sample 2 - T1CE Translation", "modality": "T1CE"},
    "visual_sample_02_T2.pdf": {"description": "Sample 2 - T2 Translation", "modality": "T2"},
    "visual_sample_02_FLAIR.pdf": {"description": "Sample 2 - FLAIR Translation", "modality": "FLAIR"},
    "visual_sample_02_all_modalities.pdf": {"description": "Sample 2 - All Modalities", "modality": "all"}
}

LATEX_TABLES = {
    "table1_quantitative_results.tex": {
        "description": "Main Quantitative Results Table",
        "data": "results/evaluation/evaluation_results.json"
    },
    "table2_cycle_consistency.tex": {
        "description": "Cycle Consistency Results Table",
        "data": "results/evaluation/cycle_consistency_results.json"
    },
    "table3_statistical_summary.tex": {
        "description": "Statistical Summary Table",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json"
    },
    "table_ablation_summary.tex": {
        "description": "Ablation Study Summary Table",
        "data": "evaluation_results/ablation/ablation_results_*.json"
    },
    "table_comprehensive_results.tex": {
        "description": "Comprehensive Results Table",
        "data": "multiple"
    },
    "table_downstream_results.tex": {
        "description": "Downstream Task Results Table",
        "data": "experiments/downstream_evaluation/evaluation_summary.json"
    },
    "table_efficiency.tex": {
        "description": "Computational Efficiency Table",
        "data": "experiments/computational_analysis/efficiency_results.json"
    },
    "table_method_comparison.tex": {
        "description": "Method Comparison Table",
        "data": "experiments/statistical_analysis/statistical_analysis_results.json"
    },
    "table_radiomics_preservation.tex": {
        "description": "Radiomics Preservation Table",
        "data": "experiments/radiomics_analysis/radiomics_preservation_results.json"
    }
}

# summary statistics
def get_inventory_summary():
    """return summary of all figures and tables"""
    return {
        "training": len(TRAINING_FIGURES),
        "ablation": len(ABLATION_FIGURES),
        "downstream": len(DOWNSTREAM_FIGURES),
        "statistical": len(STATISTICAL_FIGURES),
        "radiomics": len(RADIOMICS_FIGURES),
        "computational": len(COMPUTATIONAL_FIGURES),
        "publication": len(PUBLICATION_FIGURES),
        "architecture": len(ARCHITECTURE_FIGURES),
        "visual_examples": len(VISUAL_EXAMPLES),
        "latex_tables": len(LATEX_TABLES),
        "total_pdfs": sum([
            len(TRAINING_FIGURES),
            len(ABLATION_FIGURES),
            len(DOWNSTREAM_FIGURES),
            len(STATISTICAL_FIGURES),
            len(RADIOMICS_FIGURES),
            len(COMPUTATIONAL_FIGURES),
            len(PUBLICATION_FIGURES),
            len(ARCHITECTURE_FIGURES),
            len(VISUAL_EXAMPLES)
        ]),
        "total_tables": len(LATEX_TABLES)
    }

if __name__ == "__main__":
    summary = get_inventory_summary()
    print("neuroscope figure inventory")
    print("=" * 40)
    for category, count in summary.items():
        print(f"  {category}: {count}")
