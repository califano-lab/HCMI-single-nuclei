
### Demo codes in this folder

This folder contains Jupyter notebooks with demo codes for generating several figures in the manuscript, including **Figure 5** and **Extended Data Figure 6, 7, and 8**. To run these notebooks, please download the corresponding data and metadata from [Figshare](https://figshare.com/account/articles/28410485). Then set `Work_dir` to the downloaded `HCMI_snRNAseq_data_metadata directory` and `Code_dir` to the parent directory of this README file (`../HCMI-single-nuclei`).

-`Figure-5b-5c.ipynb`: performs malignant cell state annotation for GBM cases using signatures from [Neftel et al, 2019](https://www.sciencedirect.com/science/article/pii/S0092867419306877?via%3Dihub) and generates corresponding GBM panels for **Figures 5b** and **5c**.

-`Figure-5d.ipynb`: displays cell states in PCA-projected space from protein activity data for all GBM tumor-model pairs, including cases 416 and 002 shown in **Figure 5d**

-`Figure-5e-5f-5g-Extended-Data-Figure-7b-7e.ipynb`: performs VIPER-based malignant state annotation for GLS, MOS, ALS subtypes ([Laise et al.](https://www.biorxiv.org/content/10.1101/2020.10.27.357269v2)) in PAAD cases, and generates **Figures 5e, Ff, 5g** and **Extended Data Figure 7b** and **7e**

-`Extended-Data-Figure-6a.ipynb` and `Extended-Data-Figure-7a.ipynb`: summarize the % of malignant cells in each tumor for Glioblastoma (**Extended Data Figure 6a**) and Pancreatic Cancer (**Extended Data Figure 7a**), respectively

- `Extended-Data-Figure-6b.ipynb`, `Extended-Data-Figure-7c.ipynb` and `Extended-Data-Figure-8a.ipynb`: compute differential pathway enrichment analysis and corresponding visualizations for Glioblastoma (**Extended Data Figure 6b**), Pancreatic (**Extended Data Figure 7c**) and Colorectal Cancer (**Extended Data Figure 8a**).

- `Extended-Data-Figure-7d.ipynb`: a demonstration notebook highlighting different cellular populations within the PAAD tumor microenvironment as shown in **Extended Data Figure 7d**

- `Extended-Data-Figure-8b-8c.ipynb`: illustrates epithelial state annotation in COAD pairs using gene expression signatures from [Joanito et al., 2022](https://www.nature.com/articles/s41588-022-01100-4), and outlines the generation of **Extended Data Fig. 8b** and **8c**.

- `Demo-one-tumor-model-pair-qc-annotation.ipynb`: provides an example of a standard snRNA-seq preprocessing workflow, including quality control-based cell filtering, graph-based clustering, and cell annotation using singleR.

