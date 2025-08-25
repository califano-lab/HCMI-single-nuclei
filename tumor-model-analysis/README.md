
### Demo codes in this folder

- `all-COAD-samples-enrichment-analysis-Nature.ipynb`: illustrates how to perform epithelial state annotation in COAD pairs using gene expression signatures from [Joanito et al., 2022](https://www.nature.com/articles/s41588-022-01100-4) and outlines the generation of panels in Extended Data Fig. 8.

- `all-GBM-cohort-samples-enrichment-Nature.ipynb`: performs malignant cell state annotation for GBM cases using signatures from [Neftel et al, 2019](https://www.sciencedirect.com/science/article/pii/S0092867419306877?via%3Dihub) and generates the corresponding GBM panels in Figure 5.

- `all-GBM-cohort-samples-protein-activity-visualization-Nature.ipynb`: displays GBM cell states in PCA-projected space from protein activity data for all tumor-model pairs, including cases 416 and 002 shown in Figure 5

- `all-PDAC-samples-enrichment-analysis-Nature.ipynb`: performs VIPER-based malignant state annotation for GLS, MOS, ALS subtypes ([Laise et al.](https://www.biorxiv.org/content/10.1101/2020.10.27.357269v2)) for PAAD cases and generation of the corresponding PAAD panels in Figure 5

- `One-pair-qc-annotation.ipynb`: provides an example of a standard snRNA-seq preprocessing workflow, including quality control-based cell filtering, graph-based clustering, and cell annotation using singleR.

- `PDAC-TME-Visualization-Fig-5-Nature.ipynb`: a simple demonstration notebook that highlights different cellular populations within the PAAD tumor microenvironment

- `T-vs-M-DEG-cohort-Nature.ipynb`: runs differential pathway analysis between tumor and models across cohorts (glioblastoma, pancreas, colorectal). The default cohort is COAD.
