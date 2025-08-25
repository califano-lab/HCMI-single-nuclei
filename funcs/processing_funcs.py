import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.sparse as sp
from joblib import Parallel, delayed
import scanpy as sc
import anndata as ad
import matplotlib.colors as mcolors
import seaborn as sns
from adjustText import adjust_text
import anndata

def get_top_two_cell_types(row):
    sorted_pvals = row.sort_values()
    return pd.Series([sorted_pvals.index[0], sorted_pvals.index[1]])


def get_top_cell_types(row, top_n=2, pval_threshold=0.05):
    """
    Get the top N cell types based on p-values, with a p-value threshold check.

    Parameters:
    row (pd.Series): A pandas Series containing p-values.
    top_n (int): Number of top cell types to return.
    pval_threshold (float): p-value threshold. If the minimum p-value is above this threshold, return 'Unknown'.

    Returns:
    pd.Series: A pandas Series containing the top N cell types if the min p-value is below the threshold; otherwise, 'Unknown'.
    """
    sorted_pvals = row.sort_values()
    
    # Check if the minimum p-value is above the threshold
    if sorted_pvals.iloc[0] > pval_threshold:
        return pd.Series(['Unknown'] * top_n)
    
    # Return the top N cell types
    return pd.Series(sorted_pvals.index[:top_n])


def simplify_cell_type(cell_type_series,tissue): 
    """
    Simplify cell type annotations based on a predefined mapping.
    
    Parameters:
    - cell_type_series: pd.Series containing cell type annotations.
    
    Returns:
    - pd.Series with simplified cell type annotations.
    """
    if tissue == "Brain" or tissue == "GBM":
        conversion_table = {
            'Ependymal cells': 'Ependymal cells', ### Brain 
            'Microglia': 'Microglia',
            'Bergmann glia': 'Glial cells',
            'Neuroblasts': 'Neurons',
            'Satellite glial cells': 'Glial cells',
            'Pinealocytes': 'Endocrine cells',
            'Pyramidal cells': 'Neurons',
            'Tanycytes': 'Glial cells',
            'Astrocytes': 'Astrocytes',
            'Oligodendrocytes': 'Oligodendrocytes',
            'Oligodendrocyte progenitor cells': 'OPC',
            'Neurons': 'Neurons',
            'Serotonergic neurons': 'Neurons',
            'Noradrenergic neurons': 'Neurons',
            'Adrenergic neurons': 'Neurons',
            'Dopaminergic neurons': 'Neurons',
            'Choroid plexus cells': 'Choroid plexus cells',
            'Schwann cells': 'Glial cells',
            'Interneurons': 'Neurons',
            'Neural stem/precursor cells': 'Neural precursor cells',
            'Anterior pituitary gland cells': 'Endocrine cells',
            'Immature neurons': 'Neurons',
            'Radial glia cells': 'Glial cells',
            'Glutaminergic neurons': 'Neurons',
            'Purkinje neurons': 'Neurons',
            'Neuroendocrine cells': 'Neuroendocrine cells',
            'Motor neurons': 'Neurons',
            'Retinal ganglion cells': 'Neurons',
            'Cajal-Retzius cells': 'Neurons',
            'Cholinergic neurons': 'Neurons',
            'Meningeal cells': 'Glial cells',
            'GABAergic neurons': 'Neurons',
            'Trigeminal neurons': 'Neurons',
            'Glycinergic neurons': 'Neurons', 
            'Plasmacytoid dendritic cells': 'Dendritic cells', ### Immune system 
            'Macrophages': 'Macrophages/Monocytes',
            'Neutrophils': 'Granulocytes',
            'Dendritic cells': 'Dendritic cells',
            'Basophils': 'Granulocytes',
            'T helper cells': 'T cells',
            'Megakaryocytes': 'Megakaryocytes',
            'Natural killer T cells': 'T cells',
            'Nuocytes': 'Nuocytes',
            'Mast cells': 'Granulocytes',
            'Plasma cells': 'Plasma cells',
            'B cells naive': 'B cells',
            'B cells memory': 'B cells',
            'Monocytes': 'Macrophages/Monocytes',
            'NK cells': 'NK cells',
            'Eosinophils': 'Granulocytes',
            'Red pulp macrophages': 'Macrophages/Monocytes',
            'Myeloid-derived suppressor cells': 'Myeloid-derived suppressor cells',
            'T cells': 'T cells',
            'Gamma delta T cells': 'T cells',
            'B cells': 'B cells',
            'T memory cells': 'T cells',
            'T regulatory cells': 'T cells',
            'T follicular helper cells': 'T cells',
            'T cytotoxic cells': 'T cells',
            'Fibroblasts': 'Fibroblasts', ### Connective tissue
            'Adipocytes': 'Adipocytes',
            'Adipocyte progenitor cells': 'Adipocytes',
            'Chondrocytes': 'Chondrocytes',
            'Stromal cells': 'Stromal cells'
        }
    elif tissue == "Lymph nodes" or tissue == "Pancreas" or tissue == "PDAC":
        conversion_table = {
            'Acinar cells': 'Acinar cells', ### Pancreas
            'Pancreatic progenitor cells': 'Pancreatic progenitor cells',
            'Epsilon cells': 'Epsilon cells',
            'Ductal cells': 'Ductal cells',
            'Beta cells': 'Beta cells',
            'Pancreatic stellate cells': 'Stellate cells',
            'Alpha cells': 'Alpha cells',
            'Gamma (PP) cells': 'Gamma (PP) cells',
            'Delta cells': 'Delta cells',
            'Peri-islet Schwann cells': 'Schwann cells',
            'Plasmacytoid dendritic cells': 'Dendritic cells', ### Immune system 
            'Macrophages': 'Macrophages/Monocytes',
            'Neutrophils': 'Granulocytes',
            'Dendritic cells': 'Dendritic cells',
            'Basophils': 'Granulocytes',
            'T helper cells': 'T cells',
            'Megakaryocytes': 'Megakaryocytes',
            'Natural killer T cells': 'T cells',
            'Nuocytes': 'Nuocytes',
            'Mast cells': 'Granulocytes',
            'Plasma cells': 'Plasma cells',
            'B cells naive': 'B cells',
            'B cells memory': 'B cells',
            'Monocytes': 'Macrophages/Monocytes',
            'NK cells': 'NK cells',
            'Eosinophils': 'Granulocytes',
            'Red pulp macrophages': 'Macrophages/Monocytes',
            'Myeloid-derived suppressor cells': 'Myeloid-derived suppressor cells',
            'T cells': 'T cells',
            'Gamma delta T cells': 'T cells',
            'B cells': 'B cells',
            'T memory cells': 'T cells',
            'T regulatory cells': 'T cells',
            'T follicular helper cells': 'T cells',
            'T cytotoxic cells': 'T cells',
            'Fibroblasts': 'Fibroblasts', ### Connective tissue
            'Adipocytes': 'Adipocytes',
            'Adipocyte progenitor cells': 'Adipocytes',
            'Chondrocytes': 'Chondrocytes',
            'Stromal cells': 'Stromal cells'
        }
    elif tissue == "COAD":
        conversion_table = {
            'Paneth cells': 'Paneth cells', # GI tract
            'Enterocytes': 'Enterocytes',  
            'Crypt cells': 'Crypt cells',  
            'Gastric chief cells': 'Gastric chief cells',  
            'Enteric glia cells': 'Enteric glia cells',  
            'Goblet cells': 'Goblet cells', 
            'Microfold cells': 'Microfold cells', 
            'Foveolar cells': 'Foveolar cells',  
            'Enteroendocrine cells': 'Enteroendocrine cells',  
            'Enteric neurons': 'Enteric neurons', 
            'Tuft cells': 'Tuft cells',  
            'Enterochromaffin cells': 'Enterochromaffin cells',  
            'Parietal cells': 'Parietal cells',  
            'Plasmacytoid dendritic cells': 'Dendritic cells', ### Immune system 
            'Macrophages': 'Macrophages/Monocytes',
            'Neutrophils': 'Granulocytes',
            'Dendritic cells': 'Dendritic cells',
            'Basophils': 'Granulocytes',
            'T helper cells': 'T cells',
            'Megakaryocytes': 'Megakaryocytes',
            'Natural killer T cells': 'T cells',
            'Nuocytes': 'Nuocytes',
            'Mast cells': 'Granulocytes',
            'Plasma cells': 'Plasma cells',
            'B cells naive': 'B cells',
            'B cells memory': 'B cells',
            'Monocytes': 'Macrophages/Monocytes',
            'NK cells': 'NK cells',
            'Eosinophils': 'Granulocytes',
            'Red pulp macrophages': 'Macrophages/Monocytes',
            'Myeloid-derived suppressor cells': 'Myeloid-derived suppressor cells',
            'T cells': 'T cells',
            'Gamma delta T cells': 'T cells',
            'B cells': 'B cells',
            'T memory cells': 'T cells',
            'T regulatory cells': 'T cells',
            'T follicular helper cells': 'T cells',
            'T cytotoxic cells': 'T cells',
            'Fibroblasts': 'Fibroblasts', ### Connective tissue
            'Adipocytes': 'Adipocytes',
            'Adipocyte progenitor cells': 'Adipocytes',
            'Chondrocytes': 'Chondrocytes',
            'Stromal cells': 'Stromal cells'            
        }
    elif tissue == "PA_subtype_and_TME":
        conversion_table = {
            'ALS': 'ALS',
            'GLS': 'GLS',
            'MOS': 'MOS',
            'Unknown': 'Unknown',
            'B-cells': 'Immune',
            'DC': 'Immune',
            'Fibroblasts': 'Connective Tissue',  # Fibroblasts retain their label
            'Macrophages': 'Immune',
            'Mesangial cells': 'Connective Tissue',  # Grouped under connective role
            'Monocytes': 'Immune',
            'CD8+ T-cells': 'Immune',
            'Melanocytes': 'Others',
            'Chondrocytes': 'Connective Tissue',  # Cartilage-related cells
            'Endothelial cells': 'Endothelial',
            'Neurons': 'Neuroendocrine',
            'Epithelial cells': 'Normal Epithelial',
            'Smooth muscle': 'Connective Tissue',  # Grouped with similar tissue cells
            'Skeletal muscle': 'Connective Tissue',  # Same as above
            'Keratinocytes': 'Normal Epithelial',
            'NK cells': 'Immune',
            'CD4+ T-cells': 'Immune',
            'Others': 'Others',
            'Pericytes': 'Connective Tissue',  # Support cells near blood vessels
            'Adipocytes': 'Connective Tissue',  # Fat cells grouped here
            'HSC': 'HSC',
            'Myocytes': 'Connective Tissue',  # Muscle-like cells
            'Erythrocytes': 'Erythrocytes'
        }
    else:
        print(f"No conversion table available for tissue: {tissue}")

    # Convert Series to list, map the values using conversion table, and convert back to Series
    simplified_cell_types = cell_type_series.map(conversion_table).fillna(cell_type_series)
    
    return simplified_cell_types


def compute_counts_nn(adata, top_n_neighbors=5, n_jobs=-1, normalize=None):
    """
    Compute a new layer 'counts_nn' in an AnnData object by summing gene counts of each cell 
    with the counts of its top N nearest neighbors. Optionally perform log normalization and scaling.

    Parameters:
    - adata: AnnData object
    - top_n_neighbors: int, number of top nearest neighbors to consider
    - n_jobs: int, number of CPU cores to use for parallelization (-1 uses all available cores)
    - normalize: str or None, normalization method to apply ('logdata' or 'scaleadata')

    Returns:
    - adata: AnnData object with updated layers ('counts_nn', and optionally 'logdata_nn', 'scaledata_nn')
    """
    knn_distances = adata.obsp['distances']
    top_neighbors = [knn_distances[i].indices[np.argsort(knn_distances[i].data)[:top_n_neighbors]] for i in range(knn_distances.shape[0])]

    def sum_neighbors(i):
        nn_indices = top_neighbors[i]
        summed_counts = adata.X[i] + adata.X[nn_indices].sum(axis=0)
        return sp.csr_matrix(summed_counts)

    counts_nn = Parallel(n_jobs=n_jobs, backend='threading')(delayed(sum_neighbors)(i) for i in range(adata.shape[0]))

    # Stack the results into a single sparse matrix
    adata.layers['counts_nn'] = sp.vstack(counts_nn)

    if normalize in ['logdata', 'scaledata']:
        del counts_nn
        # Perform log normalization on the 'counts_nn' layer
        counts_nn = adata.layers['counts_nn'].copy()
        sc.pp.normalize_total(adata, target_sum=1e4, layer='counts_nn')
        sc.pp.log1p(adata, layer='counts_nn')
        adata.layers['logdata_nn'] = adata.layers['counts_nn'].copy()
        adata.layers['counts_nn'] = counts_nn  # Restore the original counts_nn

    if normalize == 'scaledata':
        del counts_nn
        # Perform scaling on the 'counts_nn' layer
        logdata_nn = adata.layers['logdata_nn'].copy()
        sc.pp.scale(adata, max_value=10, layer='logdata_nn')
        adata.layers['scaledata_nn'] = adata.layers['logdata_nn'].copy()
        adata.layers['logdata_nn'] = logdata_nn  # Restore the original counts_nn

    return adata




def generate_pie_charts(adata_samples_obs, samples, reference, figures_savings_path, case, extra_label=''):
    """
    Generate pie charts for cell type distributions and save them to a PDF.

    Parameters:
    adata_samples (dict): Dictionary of AnnData.obs objects, each corresponding to a sample.
    samples (list): List of sample names.
    reference (str): Column name in obs used for cell type annotations.
    figures_savings_path (str): Path to save the PDF file.
    case (str): Case identifier for the plot titles and file names.
    extra_label (str): Label to attach 
    """
    tmp_n_cells = []  # list to collect the number of cells per type per sample
    # Compute value counts for each sample and collect in the list
    for sample in samples:
        counts = adata_samples_obs[sample][reference].value_counts().to_dict()  # number of cells per type
        counts['sample'] = sample  # sample name to the dictionary
        tmp_n_cells.append(counts)  # append dict to list
    df_n_cells = pd.DataFrame(tmp_n_cells)  # list of dict to DataFrame

    df_n_cells.fillna(0, inplace=True)  # if any cell type is missing in a sample, it should have count 0
    cols = ['sample'] + [col for col in df_n_cells.columns if col != 'sample']  # sample to first position
    df_n_cells = df_n_cells[cols]  # df with n_cells
    df_perc_cells = df_n_cells.set_index('sample').div(df_n_cells.set_index('sample').sum(axis=1), axis=0) * 100  # df with % cells

    palette = plt.get_cmap("tab20").colors
    with PdfPages(f'{figures_savings_path}{case}_{reference}_piechart{extra_label}.pdf') as pdf:  # saving multiple plots in one pdf
        num_samples = len(samples)  # figure with a subplot for each sample
        fig, axs = plt.subplots(1, num_samples, figsize=(15, 5), subplot_kw=dict(aspect="equal"))
        legend_entries = {}  # dict to collect all unique labels and colors
        # Plot each sample as a pie chart
        for i, sample in enumerate(samples):
            data = df_perc_cells.loc[sample]
            labels = data.index
            sizes = data.values
            colors = palette[:len(labels)]
            # Filter out segments smaller than 5%
            large_segments = sizes >= 5
            small_segments = sizes < 5

            def autopct(pct):
                return ('%1.1f%%' % pct) if pct >= 5 else ''

            # Plot the pie chart
            wedges, texts, autotexts = axs[i].pie(
                sizes,
                labels=[label if size >= 5 else '' for label, size in zip(labels, sizes)],
                colors=colors,
                autopct=autopct,
                startangle=90
            )
            t_or_m = "Tumor" if adata_samples_obs[sample]["Tumor or Model"][0] == "T" else "Model" if adata_samples_obs[sample]["Tumor or Model"][0] == "M" else "Unknown"
            subplt_title = f"{t_or_m}, {adata_samples_obs[sample]['Sample Type'][0]}\n(n = {adata_samples_obs[sample].shape[0]})"
            axs[i].set_title(subplt_title)

            # Add all labels and colors to the legend_entries dictionary
            for label, color in zip(labels, colors):
                legend_entries[label] = color

        # Adjust layout and save the pie charts plot to PDF
        plt.suptitle(f"{case} - {adata_samples_obs[sample]['Notes'][0]}", fontsize=20, y=0.95)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Create a figure for the legend
        fig_legend = plt.figure(figsize=(8, 2))  # Adjust the size as needed
        ax_legend = fig_legend.add_subplot(111)

        # Create legend plot with all cell types
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markersize=10, markerfacecolor=color)
                   for label, color in legend_entries.items()]
        ax_legend.legend(handles=handles, loc='center', ncol=4)  # Adjust ncol for the number of columns
        ax_legend.axis('off')

        # Save the legend plot to PDF
        plt.tight_layout()
        pdf.savefig(fig_legend)
        plt.close(fig_legend)

def create_anndata_from_lfc(dedf):
    """
    Creates an AnnData object from a DataFrame containing log-fold changes.
    
    Parameters:
    dedf (pd.DataFrame): A DataFrame with columns 'group', 'names', and 'logfoldchanges'.
    
    Returns:
    ad.AnnData: An AnnData object with genes as variables and groups as observations.
    """
    # Step 1: Pivot the DataFrame to create a log-fold change matrix
    lfc_matrix_df = dedf.pivot(index='names', columns='group', values='logfoldchanges')

    # Step 2: Create the AnnData object with the LFC matrix
    adata_lfc = ad.AnnData(X=lfc_matrix_df.T.values)  # Transpose the matrix

    # Step 3: Assign gene names as variable names (columns) and group names as observation names (rows)
    adata_lfc.var_names = lfc_matrix_df.index  # Gene names
    adata_lfc.obs_names = lfc_matrix_df.columns  # Group names (samples)

    return adata_lfc



def process_clusters(clusters_putative_malignant):
    # Extract the value from the Series (assuming a single element in the Series)
    cluster_value = clusters_putative_malignant.values[0]
    
    # Check if the value is "all"
    if isinstance(cluster_value, str) and cluster_value.lower() == "all":
        return cluster_value
    # If it's a comma-separated string, split it into a list of integers
    elif isinstance(cluster_value, str) and "," in cluster_value:
        return [int(i) for i in cluster_value.split(",")]
    # If it's a single integer, wrap it in a list
    else:
        return [int(cluster_value)]
    

def apply_consistent_palette(adata_samples, columns_of_interest, palette="tab20"):
    """
    Apply a consistent color palette to the specified columns across all samples in adata_samples.
    
    Parameters:
    - adata_samples (dict): Dictionary of AnnData objects, one per sample.
    - columns_of_interest (list): List of column names to apply color palettes to.
    - palette (str): Color palette name for seaborn (default is "tab20").
    
    Returns:
    - None. Color palettes are applied to the `.uns` attribute of each AnnData object.
    """
    for col in columns_of_interest:
        # Collect all unique categories across all samples for the given column
        all_categories = set()
        for sample in adata_samples:
            if col in adata_samples[sample].obs.columns:
                all_categories.update(adata_samples[sample].obs[col].cat.categories)

        # Create a consistent color palette for the categories
        palette_colors = sns.color_palette(palette, len(all_categories))
        category_colors = dict(zip(sorted(all_categories), palette_colors))

        # Apply the color palette to each adata object
        for sample in adata_samples:
            if col in adata_samples[sample].obs.columns:
                # Map the colors to the column categories
                category_list = adata_samples[sample].obs[col].cat.categories
                adata_samples[sample].uns[f'{col}_colors'] = [mcolors.rgb2hex(category_colors[cat]) for cat in category_list]
import scanpy as sc
import seaborn as sns

def apply_consistent_palette_v2(adata, columns_of_interest, palette="tab20"):
    """
    Apply a consistent color palette across multiple specified columns for all samples or a single AnnData object.
    
    Parameters:
    - adata (dict or AnnData): Dictionary of AnnData objects or a single AnnData object.
    - columns_of_interest (list): List of columns to apply consistent color palettes to.
    - palette (str): Color palette name for seaborn (default is "tab20").
    
    Returns:
    - None. Color palettes are applied to the `.uns` attribute of each AnnData object.
    """
    if isinstance(adata, dict):
        adata_samples = adata
    else:
        adata_samples = {0: adata}  # Wrap single AnnData object in a dictionary

    for column in columns_of_interest:
        # Collect all unique categories for the current column
        all_categories = set()
        
        for sample in adata_samples:
            if column in adata_samples[sample].obs:
                all_categories.update(adata_samples[sample].obs[column].cat.categories)

        # Create a consistent color palette for all categories
        color_palette = sns.color_palette(palette, len(all_categories))
        category_colors = dict(zip(sorted(all_categories), color_palette))

        # Apply the color palette to each adata object
        for sample in adata_samples:
            if column in adata_samples[sample].obs:
                # Ensure each category is represented in obs
                existing_categories = adata_samples[sample].obs[column].cat.categories
                new_categories = sorted(all_categories.difference(existing_categories))
                
                if new_categories:
                    adata_samples[sample].obs[column] = adata_samples[sample].obs[column].cat.add_categories(new_categories)

                # Map colors to all categories, even if they have no entries
                adata_samples[sample].uns[f"{column}_colors"] = [
                    category_colors[cat] for cat in sorted(all_categories)
                ]

def apply_shared_palette(adata, columns_of_interest, palette="tab20"):
    """
    Apply a shared consistent color palette across the specified columns for all samples or a single AnnData object.
    
    Parameters:
    - adata (dict or AnnData): Dictionary of AnnData objects or a single AnnData object.
    - columns_of_interest (list): List of columns to apply consistent color palettes to across all samples.
    - palette (str): Color palette name for seaborn (default is "tab20").
    
    Returns:
    - None. Color palettes are applied to the `.uns` attribute of each AnnData object.
    """
    # Convert single AnnData object to a dictionary
    if isinstance(adata, sc.AnnData):
        adata_samples = {0: adata}
    else:
        adata_samples = adata

    # Collect all unique categories across all samples and columns
    all_categories = set()
    
    for sample in adata_samples:
        for col in columns_of_interest:
            if col in adata_samples[sample].obs.columns:
                all_categories.update(adata_samples[sample].obs[col].cat.categories)
    
    # Create a consistent color palette for all categories
    palette_colors = sns.color_palette(palette, len(all_categories))
    category_colors = dict(zip(sorted(all_categories), palette_colors))
    
    # Apply the color palette to each column of each sample
    for sample in adata_samples:
        for col in columns_of_interest:
            if col in adata_samples[sample].obs.columns:
                # Map the colors to the column categories
                category_list = adata_samples[sample].obs[col].cat.categories
                adata_samples[sample].uns[f'{col}_colors'] = [
                    mcolors.rgb2hex(category_colors[cat]) for cat in category_list
                ]


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def stacked_barplot(obs_dict, uns_dict, column="NaRnEA_assignment_cg", color_key="NaRnEA_assignment_cg_colors", percentage=True, case="", figures_savings_path=None, extra_label="", show_plot=True):
    """
    Create a horizontal stacked barplot with cell type proportions for each sample and export the legend as a separate page in the PDF.
    
    Parameters:
    - obs_dict (dict): Dictionary of obs DataFrames from AnnData objects (one per sample).
    - uns_dict (dict): Dictionary of uns dicts with color mappings.
    - column (str): Column with cell types (default: "NaRnEA_assigned_cg").
    - color_key (str): Key in the uns dict for color mapping (default: "NaRnEA_assignment_cg_colors").
    - percentage (bool): Whether to display proportions as percentages (0-100 scale).
    - case (str): Optional case name for the title.
    - figures_savings_path (str): Path to save the figure as a PDF.
    - extra_label (str): Additional label for the saved figure's name (default: "").
    - show_plot (bool): Whether to display the plot in the session (default: True).
    
    Returns:
    - A horizontal stacked barplot and optionally saves it to a PDF with a separate page for the legend.
    """
    # Reverse the order of the samples so the first is on top
    sample_order = list(obs_dict.keys())[::-1]

    # Create a figure and axis for the main barplot
    fig, ax = plt.subplots(figsize=(10, len(obs_dict) * 1.5))  # Adjust size for readability

    y_labels = []

    # Iterate through each sample
    for i, sample in enumerate(sample_order):
        # Get the cell type proportions
        cell_types = obs_dict[sample][column].value_counts(normalize=True)
        
        # Convert to percentage if specified
        if percentage:
            cell_types = cell_types * 100
        
        # Sort the cell types in descending order
        cell_types = cell_types.sort_values(ascending=False)

        # Get the color mapping
        colors = uns_dict[sample][color_key]
        color_map = dict(zip(obs_dict[sample][column].cat.categories, colors))
        
        # Get the 'Tumor or Model' value and number of cells
        tumor_or_model = obs_dict[sample]["Tumor or Model"].iloc[0]
        n_cells = obs_dict[sample].shape[0]
        y_label = f"{'Model' if tumor_or_model == 'M' else 'Tumor'}\nn={n_cells}"
        y_labels.append(y_label)
        
        # Plot each cell type with its color
        left_offset = 0
        for cell_type, proportion in cell_types.items():
            ax.barh(i, proportion, left=left_offset, color=color_map.get(cell_type, 'gray'), edgecolor='black', label=cell_type if i == 0 else "")
            left_offset += proportion
    
    # Set y-tick label for each sample
    ax.set_yticks(range(len(obs_dict)))
    ax.set_yticklabels(y_labels)

    # Customize plot appearance
    ax.set_xlabel("Proportion (%)" if percentage else "Proportion")
    ax.grid(False)  # Disable grid lines
    
    # Set title
    ax.set_title(f"{case} - {column}")

    # Adjust layout
    plt.tight_layout()

    # Save the figure and legend as a separate page in a PDF
    if figures_savings_path:
        with PdfPages(f'{figures_savings_path}{case}_{column}_barplot{extra_label}.pdf') as pdf:
            pdf.savefig(fig)  # Save the barplot
            
            # Create a new figure for the legend
            fig_legend = plt.figure(figsize=(10, 2))
            handles, labels = ax.get_legend_handles_labels()
            fig_legend.legend(handles, labels, loc='center', ncol=len(color_map), frameon=False)
            plt.axis('off')  # Remove axes for the legend page
            pdf.savefig(fig_legend)  # Save the legend page

    # Show the plot if specified
    if show_plot:
        plt.show()

def stacked_barplot_v2(obs_dict, uns_dict, column="NaRnEA_assignment_cg", color_key="NaRnEA_assignment_cg_colors", 
                    percentage=True, case="", figures_savings_path=None, extra_label="", show_plot=True):
    """
    Create a horizontal stacked barplot with cell type proportions for each sample and export the legend as a separate page in the PDF.
    """
    # Reverse the order of the samples so the first is on top
    sample_order = list(obs_dict.keys())[::-1]

    # Gather all unique cell types across all samples
    all_cell_types = pd.concat([obs_dict[sample][column] for sample in obs_dict]).unique()

    # Create a unified color mapping using the first sample's color key
    unified_colors = uns_dict[sample_order[0]][color_key]
    global_color_map = dict(zip(obs_dict[sample_order[0]][column].cat.categories, unified_colors))

    # Create a figure and axis for the main barplot
    fig, ax = plt.subplots(figsize=(10, len(obs_dict) * 1.5))  # Adjust size for readability

    y_labels = []

    # Iterate through each sample
    for i, sample in enumerate(sample_order):
        # Get the cell type proportions
        cell_types = obs_dict[sample][column].value_counts(normalize=True)

        # Convert to percentage if specified
        if percentage:
            cell_types = cell_types * 100

        # Sort the cell types in descending order
        cell_types = cell_types.reindex(all_cell_types).fillna(0)

        # Get the 'Tumor or Model' value and number of cells
        tumor_or_model = obs_dict[sample]["Tumor or Model"].iloc[0]
        n_cells = obs_dict[sample].shape[0]
        y_label = f"{'Model' if tumor_or_model == 'M' else 'Tumor'}\nn={n_cells}"
        y_labels.append(y_label)

        # Plot each cell type with its color from the global color map
        left_offset = 0
        for cell_type in all_cell_types:
            proportion = cell_types[cell_type]
            ax.barh(i, proportion, left=left_offset, color=global_color_map.get(cell_type, 'gray'), edgecolor='black', 
                    label=cell_type if i == 0 else "")
            left_offset += proportion

    # Set y-tick labels for each sample
    ax.set_yticks(range(len(obs_dict)))
    ax.set_yticklabels(y_labels)

    # Customize plot appearance
    ax.set_xlabel("Proportion (%)" if percentage else "Proportion")
    ax.grid(False)  # Disable grid lines

    # Set title
    ax.set_title(f"{case} - {column}")

    # Adjust layout
    plt.tight_layout()

    # Save the figure and legend as a separate page in a PDF
    if figures_savings_path:
        with PdfPages(f'{figures_savings_path}{case}_{column}_barplot{extra_label}.pdf') as pdf:
            pdf.savefig(fig)  # Save the barplot

            # Create a new figure for the legend
            fig_legend = plt.figure(figsize=(10, 2))
            handles, labels = ax.get_legend_handles_labels()
            fig_legend.legend(handles[:len(all_cell_types)], labels[:len(all_cell_types)], loc='center', 
                              ncol=len(all_cell_types), frameon=False)
            plt.axis('off')  # Remove axes for the legend page
            pdf.savefig(fig_legend)  # Save the legend page

    # Show the plot if specified
    if show_plot:
        plt.show()


def stacked_barplot_v3(obs_dict, uns_dict, column="NaRnEA_assignment_cg", color_key="NaRnEA_assignment_cg_colors", 
                   percentage=True, case="", figures_savings_path=None, extra_label="", show_plot=True):
    # Reverse the order of the samples so the first is on top
    sample_order = list(obs_dict.keys())[::-1]

    # Gather all unique cell types across all samples
    all_cell_types = pd.concat([obs_dict[sample][column] for sample in obs_dict]).unique()
    all_cell_types = sorted(all_cell_types)  # Sort for consistency

    # Create a unified color mapping
    color_mapping = {}
    for sample in sample_order:
        sample_colors = uns_dict[sample][color_key]
        sample_cell_types = obs_dict[sample][column].cat.categories
        for ct, color in zip(sample_cell_types, sample_colors):
            if ct not in color_mapping:
                color_mapping[ct] = color

    # Default color for any missing cell types
    default_color = 'gray'
    for ct in all_cell_types:
        if ct not in color_mapping:
            color_mapping[ct] = default_color

    # Create a figure and axis for the main barplot
    fig, ax = plt.subplots(figsize=(12, len(obs_dict) * 1.5))  # Adjust size for readability

    y_labels = []

    # Iterate through each sample
    for i, sample in enumerate(sample_order):
        # Get the cell type proportions
        cell_types = obs_dict[sample][column].value_counts(normalize=True)

        # Convert to percentage if specified
        if percentage:
            cell_types = cell_types * 100

        # Sort the cell types in descending order
        cell_types = cell_types.sort_values(ascending=False)

        # Get the 'Tumor or Model' value and number of cells
        tumor_or_model = obs_dict[sample]["Tumor or Model"].iloc[0]
        n_cells = obs_dict[sample].shape[0]
        y_label = f"{'Model' if tumor_or_model == 'M' else 'Tumor'}\nn={n_cells}"
        y_labels.append(y_label)

        # Plot each cell type in the sorted order using the global color map
        left_offset = 0
        for cell_type, proportion in cell_types.items():
            color = color_mapping.get(cell_type, default_color)
            ax.barh(i, proportion, left=left_offset, color=color, edgecolor='black')
            left_offset += proportion

    # Set y-tick labels for each sample
    ax.set_yticks(range(len(obs_dict)))
    ax.set_yticklabels(y_labels)

    # Customize plot appearance
    ax.set_xlabel("Proportion (%)" if percentage else "Proportion")
    ax.grid(False)  # Disable grid lines

    # Set title
    ax.set_title(f"{case} - {column}")

    # Adjust layout
    plt.tight_layout()

    # Show the plot if specified
    if show_plot:
        plt.show()

    # Save the figure and legend as a separate page in a PDF
    if figures_savings_path:
        with PdfPages(f'{figures_savings_path}{case}_{column}_barplot{extra_label}.pdf') as pdf:
            pdf.savefig(fig)  # Save the barplot
            plt.close(fig)  # Close the main figure

            # Create a new figure for the legend
            fig_legend = plt.figure(figsize=(12, 2))
            handles = [plt.Rectangle((0,0),1,1, color=color_mapping[ct]) for ct in all_cell_types]
            labels = list(all_cell_types)
            fig_legend.legend(handles, labels, loc='center', ncol=len(all_cell_types), frameon=False)
            plt.axis('off')  # Remove axes for the legend page
            pdf.savefig(fig_legend)  # Save the legend page
            plt.show(fig_legend)  # Show the legend figure
            plt.close(fig_legend)  # Close the legend figure



def volcano_plot(adata, group, n_genes=10, abs_logfc_thresh=5, logfc_thresholds=(-2, 2), title='Volcano Plot', 
                 savings_output_file='volcano_plot.pdf', show_plot=True, figsize=(8, 6), dot_size=3, dot_type='dot'):
    """
    Creates a volcano plot for a specific group in an AnnData object after running sc.tl.rank_genes_groups.

    Parameters:
    -----------
    adata : AnnData object
        An AnnData object containing ranked gene information after running sc.tl.rank_genes_groups.
    
    group : str
        The group or cluster name to plot the results for.
    
    n_genes : int
        The number of top genes to label on the plot (default is 10).
    
    abs_logfc_thresh : float
        The absolute threshold for trimming logFC values (default is 5).
    
    logfc_thresholds : tuple
        A tuple containing the logFC thresholds for the vertical dashed lines (default is (-2, 2)).
    
    title : str
        The title of the volcano plot (default is 'Volcano Plot').
    
    savings_output_file : str
        The path to save the plot (default is 'volcano_plot.pdf').

    show_plot : bool
        If True, the plot will be displayed (default is True).
    
    figsize : tuple
        Size of the figure (default is (8, 6)).
    
    dot_size : float
        Size of the dots in the plot (default is 3).
    
    dot_type : str
        Type of dot to plot. Options are 'dot' (filled) or 'circle' (open circle) (default is 'dot').

    Returns:
    --------
    A volcano plot for the specified group.
    """
    
    # Extract rank_genes_groups data
    gene_names = adata.uns['rank_genes_groups']['names'][group]
    logfc = adata.uns['rank_genes_groups']['logfoldchanges'][group]
    pvals_adj = adata.uns['rank_genes_groups']['pvals_adj'][group]
    
    # Replace 0 p-values with machine precision
    pvals_adj = np.where(pvals_adj == 0, np.finfo(float).tiny, pvals_adj)
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Gene': gene_names,
        'logFC': logfc,
        'adj.P.Val': pvals_adj
    })
    
    # Cap logFC values based on the absolute threshold
    df['logFC'] = np.clip(df['logFC'], -abs_logfc_thresh, abs_logfc_thresh)
    
    # Dot type customization
    marker = 'o' if dot_type == 'dot' else 'o' if dot_type == 'circle' else 'o'  # Adjust the dot style
    
    # Create a PDF file to save the plot
    with PdfPages(savings_output_file) as pdf:
        # Scatter plot for all genes
        plt.figure(figsize=figsize)
        plt.scatter(x=df['logFC'], y=-np.log10(df['adj.P.Val']), s=dot_size, color="gray", marker=marker)
        
        # Highlight down-regulated genes (logFC <= -2 and adj.P.Val <= 0.01)
        down = df[(df['logFC'] <= logfc_thresholds[0]) & (df['adj.P.Val'] <= 0.01)]
        plt.scatter(x=down['logFC'], y=-np.log10(down['adj.P.Val']), s=dot_size, color="blue", marker=marker)
        
        # Highlight up-regulated genes (logFC >= 2 and adj.P.Val <= 0.01)
        up = df[(df['logFC'] >= logfc_thresholds[1]) & (df['adj.P.Val'] <= 0.01)]
        plt.scatter(x=up['logFC'], y=-np.log10(up['adj.P.Val']), s=dot_size, color="red", marker=marker)
        
        # Label the top N genes from the up-regulated group
        texts = []
        for i, r in up.head(n_genes).iterrows():
            texts.append(plt.text(x=r['logFC'], y=-np.log10(r['adj.P.Val']), s=r['Gene'], fontsize=11))
        
        # Label the top N down-regulated genes based on adjusted p-values
        for i, r in down.tail(n_genes).iterrows():
            texts.append(plt.text(x=r['logFC'], y=-np.log10(r['adj.P.Val']), s=r['Gene'], fontsize=11))
        
        # Adjust text to avoid overlap
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))
        
        # Plot customization
        plt.xlabel("log(Fold-Change)")  # Updated x-axis label
        plt.ylabel("-log10(adjusted P value)")  # Updated y-axis label
        
        # Vertical dashed lines for logFC thresholds
        plt.axvline(logfc_thresholds[0], color="grey", linestyle="--")
        plt.axvline(logfc_thresholds[1], color="grey", linestyle="--")
        plt.axhline(2, color="grey", linestyle="--")
        
        # Remove grid
        plt.grid(False)
        
        # Title for the plot
        plt.title(title)

        # Save the figure to the PDF file
        pdf.savefig()  # saves the current figure into a pdf page
        #plt.close()    # close the figure after saving

        # Show the plot if specified
        if show_plot:
            plt.show()


import pandas as pd

def format_oncomatch_text(case_metadata):
    """
    Format case metadata into a single descriptive string.

    Parameters:
    case_metadata (pd.DataFrame): DataFrame containing case metadata with relevant columns.

    Returns:
    str: Formatted string summarizing the case metadata.
    """
    # Initialize empty variables to collect information
    case_id = case_metadata["Case ID"].iloc[0]  # Assuming all rows have the same Case ID
    sample_type = case_metadata["Sample Type"].iloc[0]  # Assuming all rows have the same Sample Type

    # Prepare lists to store OncoMatch_moma_call based on Tumor or Model
    tumor_calls = []
    model_calls = []

    for _, row in case_metadata.iterrows():
        if row["Tumor or Model"] == "T":
            tumor_calls.append(row["OncoMatch_moma_call"])
        elif row["Tumor or Model"] == "M":
            model_calls.append(row["OncoMatch_moma_call"])

    # Create the output string
    tumor_text = ', '.join(tumor_calls) if tumor_calls else "None"
    model_text = ', '.join(model_calls) if model_calls else "None"

    #result_text = f"{case_id} ({sample_type}) - Tumor: {tumor_text}; Model: {model_text}"
    result_text = f"bulk RNA-seq - Tumor: {tumor_text}; Model: {model_text}"
    
    return result_text

def compute_mean_by_group(adata: anndata.AnnData, group_col: str) -> anndata.AnnData:
    # Step 1: Convert obs to a DataFrame
    obs_df = adata.obs.copy()
    
    # Step 2: Convert var to a DataFrame
    var_df = adata.var.copy()
    
    # Step 3: Create a DataFrame for the var (scores) to compute the mean
    var_scores_df = pd.DataFrame(adata.X, index=obs_df.index, columns=var_df.index)
    
    # Step 4: Group by the specified column and calculate the mean for each variable
    mean_scores = var_scores_df.groupby(obs_df[group_col]).mean()
    
    # Reset index to make the grouping column a column in obs, not the index
    mean_scores = mean_scores.reset_index()
    
    # Step 5: Create a new AnnData object with the mean scores
    adata_mean = anndata.AnnData(
        X=mean_scores.drop(columns=[group_col]).values,
        obs=mean_scores[[group_col]],  # Keep the grouping column in obs
        var=pd.DataFrame(index=mean_scores.columns.drop(group_col))
    )
    
    # Set obs_names and var_names
    adata_mean.obs_names = mean_scores.index.astype(str)
    adata_mean.var_names = mean_scores.columns.drop(group_col)

    return adata_mean


def stacked_barplot_paired(obs_dict, uns_dict, column="NaRnEA_assignment_cg", color_key="NaRnEA_assignment_cg_colors", 
                   percentage=True, case="", figures_savings_path=None, extra_label="", show_plot=True):
    # Reverse the order of the samples so the first is on top
    sample_order = list(obs_dict.keys())[::-1]

    # Gather all unique cell types across all samples
    all_cell_types = pd.concat([obs_dict[sample][column] for sample in obs_dict]).unique()
    # Sort for consistency while preserving the order of first appearance
    sorted_cell_types = sorted(all_cell_types)

    # Create a unified color mapping
    color_mapping = {}
    for sample in sample_order:
        sample_colors = uns_dict[sample][color_key]
        sample_cell_types = obs_dict[sample][column].cat.categories
        for ct, color in zip(sample_cell_types, sample_colors):
            if ct not in color_mapping:
                color_mapping[ct] = color

    # Default color for any missing cell types
    default_color = 'gray'
    for ct in sorted_cell_types:
        if ct not in color_mapping:
            color_mapping[ct] = default_color

    # Create a figure and axis for the main barplot
    fig, ax = plt.subplots(figsize=(12, len(obs_dict) * 1.5))  # Adjust size for readability

    y_labels = []

    # Iterate through each sample
    for i, sample in enumerate(sample_order):
        # Get the cell type proportions
        cell_types = obs_dict[sample][column].value_counts(normalize=True)

        # Convert to percentage if specified
        if percentage:
            cell_types = cell_types * 100

        # Sort the cell types according to sorted_cell_types order
        cell_types = cell_types.reindex(sorted_cell_types, fill_value=0)

        # Get the 'Tumor or Model' value and number of cells
        tumor_or_model = obs_dict[sample]["Tumor or Model"].iloc[0]
        n_cells = obs_dict[sample].shape[0]
        y_label = f"{'Model' if tumor_or_model == 'M' else 'Tumor'}\nn={n_cells}"
        y_labels.append(y_label)

        # Plot each cell type in the sorted order using the global color map
        left_offset = 0
        for cell_type in sorted_cell_types:
            proportion = cell_types[cell_type]
            color = color_mapping.get(cell_type, default_color)
            ax.barh(i, proportion, left=left_offset, color=color, edgecolor='black')
            left_offset += proportion

    # Set y-tick labels for each sample
    ax.set_yticks(range(len(obs_dict)))
    ax.set_yticklabels(y_labels)

    # Customize plot appearance
    ax.set_xlabel("Proportion (%)" if percentage else "Proportion")
    ax.grid(False)  # Disable grid lines

    # Set title
    ax.set_title(f"{case} - {column}")

    # Adjust layout
    plt.tight_layout()

    # Show the plot if specified
    if show_plot:
        plt.show()

    # Save the figure and legend as a separate page in a PDF
    if figures_savings_path:
        with PdfPages(f'{figures_savings_path}{case}_{column}_barplot{extra_label}.pdf') as pdf:
            pdf.savefig(fig)  # Save the barplot
            plt.close(fig)  # Close the main figure

            # Create a new figure for the legend
            fig_legend = plt.figure(figsize=(12, 2))
            handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[ct]) for ct in sorted_cell_types]
            labels = list(sorted_cell_types)
            fig_legend.legend(handles, labels, loc='center', ncol=len(sorted_cell_types), frameon=False)
            plt.axis('off')  # Remove axes for the legend page
            pdf.savefig(fig_legend)  # Save the legend page
            plt.show(fig_legend)  # Show the legend figure
            plt.close(fig_legend)  # Close the legend figure



def sort_anndata_custom_var(
    adata: anndata.AnnData,
    obs_col: str,            # Observation column for grouping (e.g., 'Sample ID snRNAseq')
    var_names: list,        # List of variable names to sort by (e.g., ['Classical', 'Mesenchymal', 'Proneural'])
    ascending_order: list    # List of booleans for ascending order for each variable
) -> anndata.AnnData:
    """
    Sorts an AnnData object based on specified observation and variable columns.

    Parameters:
    - adata: AnnData object to sort.
    - obs_col: Observation column to group by (e.g., 'Sample ID snRNAseq').
    - var_names: List of variable names to sort by (e.g., ['Classical', 'Mesenchymal', 'Proneural']).
    - ascending_order: List of booleans indicating ascending order for each variable.

    Returns:
    - A sorted AnnData object.
    """
    # Step 1: Extract the values for the specified variables
    variable_indices = [adata.var_names.get_loc(var) for var in var_names]
    variable_values = [adata.X[:, idx].flatten() for idx in variable_indices]

    # Step 2: Get the categorical values of the observation column
    obs_values = adata.obs[obs_col].values

    # Step 3: Create a DataFrame for sorting
    # Combine the values into a DataFrame for easy sorting
    sort_df = pd.DataFrame({
        obs_col: obs_values,
        **{var: values for var, values in zip(var_names, variable_values)}
    })

    # Step 4: Sort the DataFrame by the observation column and then the variables
    sorted_df = sort_df.sort_values(by=[obs_col] + var_names, 
                                     ascending=[True] + ascending_order)

    # Step 5: Get the sorted indices
    sorted_indices = sorted_df.index

    # Step 6: Reorder the AnnData object using the sorted indices
    adata_sorted = adata[sorted_indices, :]

    return adata_sorted


def sort_anndata_custom_var_obs(
    adata: anndata.AnnData,
    obs_cols: list,            # List of observation columns for grouping
    var_names: list,          # List of variable names to sort by
    ascending_order: list      # List of booleans indicating ascending order for obs and var columns
) -> anndata.AnnData:
    """
    Sorts an AnnData object based on specified observation and variable columns.

    Parameters:
    - adata: AnnData object to sort.
    - obs_cols: List of observation columns to group by.
    - var_names: List of variable names to sort by.
    - ascending_order: List of booleans indicating ascending order for obs and var columns.

    Returns:
    - A sorted AnnData object.
    """
    # Step 1: Extract the values for the specified variables
    variable_indices = [adata.var_names.get_loc(var) for var in var_names]
    variable_values = [adata.X[:, idx].flatten() for idx in variable_indices]

    # Step 2: Get the categorical values of the observation columns
    obs_values = [adata.obs[col].values for col in obs_cols]

    # Step 3: Create a DataFrame for sorting
    # Combine the values into a DataFrame for easy sorting
    sort_df = pd.DataFrame({
        **{col: obs_values[i] for i, col in enumerate(obs_cols)},
        **{var: values for var, values in zip(var_names, variable_values)}
    })

    # Step 4: Sort the DataFrame by the observation columns and then the variables
    sorted_df = sort_df.sort_values(by=obs_cols + var_names, 
                                     ascending=ascending_order)

    # Step 5: Get the sorted indices
    sorted_indices = sorted_df.index

    # Step 6: Reorder the AnnData object using the sorted indices
    adata_sorted = adata[sorted_indices, :]

    return adata_sorted




def plot_stacked_horizontal_barplot(adata_obs, colormap_name, output_file, figure_title, show=True):
    # Grouping and counting cells per sample and PA_subtype_assignment
    counts = adata_obs.groupby(['Sample ID snRNAseq', 'PA_subtype_assignment']).size().unstack(fill_value=0)

    # Calculating the total number of cells per sample
    total_cells_per_sample = counts.sum(axis=1)

    # Calculating the proportions (in %) for each PA_subtype_assignment
    proportions = counts.div(total_cells_per_sample, axis=0) * 100

    # Preserving the categorical order for 'Sample ID snRNAseq'
    sample_order = adata_obs['Sample ID snRNAseq'].cat.categories

    # Creating labels for the y-axis with specified format for each sample
    new_labels = []
    for sample in sample_order:
        tumor_or_model = 'Tumor' if adata_obs['Tumor or Model'][adata_obs['Sample ID snRNAseq'] == sample].iloc[0] == 'T' else 'Model'
        sample_type = adata_obs['Sample Type'][adata_obs['Sample ID snRNAseq'] == sample].iloc[0].replace('Frozen-', '')
        new_labels.append(f"{tumor_or_model}, {sample_type}")  # Only the main label

    # Create indices for the pairs (attached to each other)
    num_samples = len(sample_order)
    pairwise_indices = np.arange(num_samples)  # No gaps between pairs

    # Creating the stacked bar plot with pairwise grouping
    fig, ax = plt.subplots(figsize=(10, 7))

    # Get the colormap
    colormap = plt.cm.get_cmap(colormap_name, len(proportions.columns))

    # Loop through each subtype and plot for each sample with a black edge color
    for idx, subtype in enumerate(proportions.columns):
        ax.barh(pairwise_indices, proportions.loc[sample_order, subtype],
                 left=proportions.loc[sample_order].cumsum(axis=1).shift(1, axis=1).fillna(0)[subtype], 
                 label=subtype, edgecolor='black', color=colormap(idx))

    # Adding the new sample labels on the y-axis
    ax.set_yticks(pairwise_indices)
    ax.set_yticklabels(new_labels)  # Only main labels without count for y-ticks

    # Invert the y-axis to have the first sample at the top
    ax.invert_yaxis()

    # Adding labels and title
    plt.xlabel('Proportion (%)')
    plt.ylabel('')  # Remove the y-axis title
    plt.title(figure_title)

    # Removing the grid
    ax.grid(False)

    # Adding the labels for case information between pairs
    for i in range(0, len(sample_order), 2):  # Iterate in steps of 2
        case_id = adata_obs['Case ID'][adata_obs['Sample ID snRNAseq'] == sample_order[i]].iloc[0]
        ax.text(-25, i + 0.5, case_id, va='center', ha='center', fontsize=14, fontweight='bold')  # Adjust x-position more to the left

    # Calculate maximum bar width for dynamic positioning
    max_width = proportions.sum(axis=1).max()  # Maximum total proportion

    # Adding the counts inline to the right of the y-labels, just after the bars
    for i, sample in enumerate(sample_order):
        count_text = f"(n={int(total_cells_per_sample[sample])})"
        ax.text(max_width + 2, i, count_text, va='center', ha='left', fontsize=12, color='black')  # Adjust x-position based on max_width

    # Adding the legend for subtypes below the plots, without title
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(proportions.columns))

    # Adjusting layout
    plt.tight_layout()

    # Save the plot to a PDF file
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)

    # Show the figure if the parameter is set to True
    if show:
        plt.show()

    plt.close(fig)


def plot_stacked_horizontal_barplot_fixed(adata_obs, fixed_colormap, output_file, figure_title, show=True):
    # Grouping and counting cells per sample and PA_subtype_assignment
    counts = adata_obs.groupby(['Sample ID snRNAseq', 'PA_subtype_assignment']).size().unstack(fill_value=0)

    # Calculating the total number of cells per sample
    total_cells_per_sample = counts.sum(axis=1)

    # Calculating the proportions (in %) for each PA_subtype_assignment
    proportions = counts.div(total_cells_per_sample, axis=0) * 100

    # Preserving the categorical order for 'Sample ID snRNAseq'
    sample_order = adata_obs['Sample ID snRNAseq'].cat.categories

    # Creating labels for the y-axis with specified format for each sample
    new_labels = []
    for sample in sample_order:
        tumor_or_model = 'Tumor' if adata_obs['Tumor or Model'][adata_obs['Sample ID snRNAseq'] == sample].iloc[0] == 'T' else 'Model'
        sample_type = adata_obs['Sample Type'][adata_obs['Sample ID snRNAseq'] == sample].iloc[0].replace('Frozen-', '')
        new_labels.append(f"{tumor_or_model}, {sample_type}")  # Only the main label

    # Create indices for the pairs (attached to each other)
    num_samples = len(sample_order)
    pairwise_indices = np.arange(num_samples)  # No gaps between pairs

    # Creating the stacked bar plot with pairwise grouping
    fig, ax = plt.subplots(figsize=(10, 7))

    # Loop through each subtype and plot for each sample with a black edge color
    for subtype in proportions.columns:
        color = fixed_colormap.get(subtype, 'gray')  # Default to gray if subtype not found
        ax.barh(pairwise_indices, proportions.loc[sample_order, subtype],
                 left=proportions.loc[sample_order].cumsum(axis=1).shift(1, axis=1).fillna(0)[subtype], 
                 label=subtype, edgecolor='black', color=color)

    # Adding the new sample labels on the y-axis
    ax.set_yticks(pairwise_indices)
    ax.set_yticklabels(new_labels)  # Only main labels without count for y-ticks

    # Invert the y-axis to have the first sample at the top
    ax.invert_yaxis()

    # Adding labels and title
    plt.xlabel('Proportion (%)')
    plt.ylabel('')  # Remove the y-axis title
    plt.title(figure_title)

    # Removing the grid
    ax.grid(False)

    # Adding the labels for case information between pairs
    for i in range(0, len(sample_order), 2):  # Iterate in steps of 2
        case_id = adata_obs['Case ID'][adata_obs['Sample ID snRNAseq'] == sample_order[i]].iloc[0]
        ax.text(-25, i + 0.5, case_id, va='center', ha='center', fontsize=14, fontweight='bold')  # Adjust x-position more to the left

    # Calculate maximum bar width for dynamic positioning
    max_width = proportions.sum(axis=1).max()  # Maximum total proportion

    # Adding the counts inline to the right of the y-labels, just after the bars
    for i, sample in enumerate(sample_order):
        count_text = f"(n={int(total_cells_per_sample[sample])})"
        ax.text(max_width + 2, i, count_text, va='center', ha='left', fontsize=12, color='black')  # Adjust x-position based on max_width

    # Adding the legend for subtypes below the plots, without title
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(proportions.columns))

    # Adjusting layout
    plt.tight_layout()

    # Save the plot to a PDF file
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig)

    # Show the figure if the parameter is set to True
    if show:
        plt.show()

    plt.close(fig)

def create_adata_to_heatmap(adata, subtypes, subtype_assignments, binary_output=False):
    """
    Create a new AnnData object with values masked based on subtype assignments.
    
    Parameters:
    - adata: AnnData object to process.
    - subtypes: list indicating the subtypes that should be in .var
    - subtype_assignments: pd.Series indicating the subtype for each observation in adata.obs.
    - binary_output: bool, if True, masked values are set to 1 instead of retaining original values.
    
    Returns:
    - adata: masked AnnData object.
    """
    
    # Step 1: Mask values based on subtype
    for subtype in subtypes:
        # Mask rows where the subtype matches
        mask = subtype_assignments == subtype
        
        # Create a column of NaNs for all entries that don't match the subtype
        non_matching_vars = adata.var_names[adata.var_names != subtype]
        
        if (binary_output == True):
            adata[mask, subtype].X = 1  # Set matching subtype values to 1

        # Mask non-matching subtype columns by setting values to NaN
        for var in non_matching_vars:
            adata[mask, var] = np.nan


    return adata


# Helper function to plot each car on the radar chart.
def add_to_radar(ax, dataframe,sample, color, angles, linecolor='black', linewidth=1.5, alpha=0.5):
  values = dataframe.loc[sample].tolist()
  values += values[:1]
  ax.plot(angles, values, color=linecolor, linewidth=linewidth, label=sample)
  ax.fill(angles, values, color=color, alpha=alpha)


def simplify_identifier(identifier):
    return identifier.split('-')[2][1:]  # Extracts the middle part and removes the first digit


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_grouped_stacked_bar(
    df, 
    figsize=(6, 6), 
    title="Stacked Bar Plot for Subtypes", 
    colormap=None, 
    group_labels=None, 
    group_gap=0.5, 
    tick_label_size=12, 
    title_size=14, 
    legend_size=12, 
    axis_label_size=12, 
    output_path=None,
    n_cells_dataframe=None  # Optional dataframe with cell counts
):
    """
    Plots a grouped stacked bar chart and optionally saves it to a PDF.

    Parameters:
    - df (pd.DataFrame): Data for the bar plot.
    - figsize (tuple): Size of the figure.
    - title (str): Title of the plot.
    - colormap (dict): Colormap for the subtypes.
    - group_labels (list): Labels for the groups.
    - group_gap (float): Gap between groups.
    - tick_label_size (int): Font size for tick labels.
    - title_size (int): Font size for the title.
    - legend_size (int): Font size for the legend.
    - axis_label_size (int): Font size for axis labels.
    - output_path (str): Path to save the plot as a PDF using PdfPages.
    - n_cells_dataframe (pd.DataFrame): Optional dataframe containing the number of cells for each sample.
    """
    # Number of groups (pair of samples)
    n_samples = len(df)
    n_groups = n_samples // 2
    
    # Define the width of the individual bars
    bar_width = 0.4  # Width of individual bars

    # Calculate positions for each group of bars
    positions = []
    for group in range(n_groups):
        positions.append([group * (bar_width * 2 + group_gap), group * (bar_width * 2 + group_gap) + bar_width])

    # Flatten the list for individual bars positions
    positions = [p for group in positions for p in group]

    # Use default colormap if none is provided
    if colormap is None:
        colormap = {
            'MES': '#AEC7E8',
            'AC': '#1F77B4',
            'NPC': '#98DF8A',
            'OPC': '#9467BD'
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars
    bottoms = np.zeros(n_samples)  # Initialize bottom positions for stacking
    for subtype in df.columns:
        ax.bar(positions, df[subtype], width=bar_width, bottom=bottoms, label=subtype, color=colormap[subtype], edgecolor='black')
        bottoms += df[subtype]  # Update bottom positions for stacking

    # Adjust x-axis (group labels at the center of each pair)
    group_positions = [sum(positions[i:i+2]) / 2 for i in range(0, len(positions), 2)]  # Midpoint of each group
    ax.set_xticks(group_positions)  # Midpoint of each group

    # Set custom group labels if provided, otherwise default to 'Group 1', 'Group 2', etc.
    if group_labels:
        ax.set_xticklabels(group_labels, fontsize=tick_label_size, fontweight="bold", ha="center", va="bottom")  # Bold text at the top
    else:
        ax.set_xticklabels([f"Group {i+1}" for i in range(n_groups)], fontsize=tick_label_size, fontweight="bold", ha="center", va="bottom")  # Default group labels

    # Add "Tumor" and "Model" labels below each bar (small font size)
    for i in range(n_groups):
        tumor_pos = positions[i * 2]  # Position of the first bar (Tumor)
        model_pos = positions[i * 2 + 1]  # Position of the second bar (Model)
        
        # Add "Tumor" and "Model" text with cell counts
        tumor_sample = df.index[i * 2]  # Sample name for tumor
        model_sample = df.index[i * 2 + 1]  # Sample name for model
        
        tumor_n_cells = n_cells_dataframe.loc[n_cells_dataframe['sample'] == tumor_sample, 'n_cells'].values[0] if n_cells_dataframe is not None else None
        model_n_cells = n_cells_dataframe.loc[n_cells_dataframe['sample'] == model_sample, 'n_cells'].values[0] if n_cells_dataframe is not None else None
        
        ax.text(tumor_pos, -3, f"Tumor\nn={tumor_n_cells}", ha='center', va='top', fontsize=7)  # "Tumor" text below with cell count
        ax.text(model_pos, -3, f"Model\nn={model_n_cells}", ha='center', va='top', fontsize=7)  # "Model" text below with cell count

    # Add group labels at the top of the figure (near y=100)
    for i, label in enumerate(group_labels if group_labels else [f"Group {i+1}" for i in range(n_groups)]):
        group_pos = group_positions[i]
        ax.text(group_pos, 102, label, ha='center', va='bottom', fontsize=tick_label_size, fontweight='bold')  # Add labels at y=100 height

    # Adjust y-axis tick label size
    ax.tick_params(axis='y', labelsize=tick_label_size)

    # Remove the grid
    ax.grid(False)

    # Set y-axis limit
    ax.set_ylim(0, 100)

    # Set x-axis limit to start from 0 and end with the last group
    ax.set_xlim(min(positions) - bar_width / 2, max(positions) + bar_width / 2)

    # Customize plot
    ax.set_ylabel("Proportion (%)", fontsize=axis_label_size)
    ax.set_xlabel("", fontsize=axis_label_size)
    ax.set_title(title, fontsize=title_size)

    # Place the legend at the bottom, centered, with horizontal alignment
    ax.legend(
        title="", 
        loc="lower center", 
        bbox_to_anchor=(0.5, -0.5),  # Center the legend below the plot
        ncol=len(df.columns),         # Make the legend horizontal with one column per subtype
        fontsize=legend_size          # Customize legend font size
    )

    # Remove x-tick labels and x-tick marks from the bottom
    ax.set_xticklabels([])  # Remove labels
    ax.tick_params(axis='x', which='both', bottom=False)  # Remove the ticks on the bottom

    plt.tight_layout()

    # Save to PDF if output path is provided
    if output_path:
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            print(f"Plot saved to {output_path}")

    # Show the plot
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_grouped_stacked_bar(
    df, 
    figsize=(6, 6), 
    title="Stacked Bar Plot for Subtypes", 
    colormap=None, 
    group_labels=None, 
    group_gap=0.5, 
    tick_label_size=12, 
    title_size=14, 
    legend_size=12, 
    axis_label_size=12, 
    output_path=None,
    n_cells_dataframe=None  # Optional dataframe with cell counts
):
    """
    Plots a grouped stacked bar chart and optionally saves it to a PDF.

    Parameters:
    - df (pd.DataFrame): Data for the bar plot.
    - figsize (tuple): Size of the figure.
    - title (str): Title of the plot.
    - colormap (dict): Colormap for the subtypes.
    - group_labels (list): Labels for the groups.
    - group_gap (float): Gap between groups.
    - tick_label_size (int): Font size for tick labels.
    - title_size (int): Font size for the title.
    - legend_size (int): Font size for the legend.
    - axis_label_size (int): Font size for axis labels.
    - output_path (str): Path to save the plot as a PDF using PdfPages.
    - n_cells_dataframe (pd.DataFrame): Optional dataframe containing the number of cells for each sample.
    """
    # Number of groups (pair of samples)
    n_samples = len(df)
    n_groups = n_samples // 2
    
    # Define the width of the individual bars
    bar_width = 0.4  # Width of individual bars

    # Calculate positions for each group of bars
    positions = []
    for group in range(n_groups):
        positions.append([group * (bar_width * 2 + group_gap), group * (bar_width * 2 + group_gap) + bar_width])

    # Flatten the list for individual bars positions
    positions = [p for group in positions for p in group]

    # Use default colormap if none is provided
    if colormap is None:
        colormap = {
            'MES': '#AEC7E8',
            'AC': '#1F77B4',
            'NPC': '#98DF8A',
            'OPC': '#9467BD'
        }

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars
    bottoms = np.zeros(n_samples)  # Initialize bottom positions for stacking
    for subtype in df.columns:
        ax.bar(positions, df[subtype], width=bar_width, bottom=bottoms, label=subtype, color=colormap[subtype], edgecolor='black')
        bottoms += df[subtype]  # Update bottom positions for stacking

    # Adjust x-axis (group labels at the center of each pair)
    group_positions = [sum(positions[i:i+2]) / 2 for i in range(0, len(positions), 2)]  # Midpoint of each group
    ax.set_xticks(group_positions)  # Midpoint of each group

    # Set custom group labels if provided, otherwise default to 'Group 1', 'Group 2', etc.
    if group_labels:
        ax.set_xticklabels(group_labels, fontsize=tick_label_size, fontweight="bold", ha="center", va="bottom")  # Bold text at the top
    else:
        ax.set_xticklabels([f"Group {i+1}" for i in range(n_groups)], fontsize=tick_label_size, fontweight="bold", ha="center", va="bottom")  # Default group labels

    # Add "Tumor" and "Model" labels below each bar (small font size)
    for i in range(n_groups):
        tumor_pos = positions[i * 2]  # Position of the first bar (Tumor)
        model_pos = positions[i * 2 + 1]  # Position of the second bar (Model)
        
        # Add "Tumor" and "Model" text with cell counts
        tumor_sample = df.index[i * 2]  # Sample name for tumor
        model_sample = df.index[i * 2 + 1]  # Sample name for model
        
        tumor_n_cells = n_cells_dataframe.loc[n_cells_dataframe['sample'] == tumor_sample, 'n_cells'].values[0] if n_cells_dataframe is not None else None
        model_n_cells = n_cells_dataframe.loc[n_cells_dataframe['sample'] == model_sample, 'n_cells'].values[0] if n_cells_dataframe is not None else None
        
        #ax.text(tumor_pos, -3, f"Tumor\nn={tumor_n_cells}", ha='center', va='top', fontsize=7)  # "Tumor" text below with cell count
        #ax.text(model_pos, -3, f"Model\nn={model_n_cells}", ha='center', va='top', fontsize=7)  # "Model" text below with cell count

        ax.text(tumor_pos, -3, "T", ha='center', va='top', fontsize=7)  # "Tumor" text below with cell count
        ax.text(model_pos, -3, "M", ha='center', va='top', fontsize=7)  # "Model" text below with cell count


    # Add group labels at the top of the figure (near y=100)
    for i, label in enumerate(group_labels if group_labels else [f"Group {i+1}" for i in range(n_groups)]):
        group_pos = group_positions[i]
        ax.text(group_pos, 102, label, ha='center', va='bottom', fontsize=tick_label_size, fontweight='bold')  # Add labels at y=100 height

    # Adjust y-axis tick label size
    ax.tick_params(axis='y', labelsize=tick_label_size)

    # Remove the grid
    ax.grid(False)

    # Set y-axis limit
    ax.set_ylim(0, 100)

    # Set x-axis limit to start from 0 and end with the last group
    ax.set_xlim(min(positions) - bar_width / 2, max(positions) + bar_width / 2)

    # Customize plot
    ax.set_ylabel("Proportion (%)", fontsize=axis_label_size)
    ax.set_xlabel("", fontsize=axis_label_size)
    ax.set_title(title, fontsize=title_size)

    # Place the legend at the bottom, centered, with horizontal alignment
    ax.legend(
        title="", 
        loc="lower center", 
        bbox_to_anchor=(0.5, -0.35),  # Center the legend below the plot
        ncol=len(df.columns),         # Make the legend horizontal with one column per subtype
        fontsize=legend_size          # Customize legend font size
    )

    # Remove x-tick labels and x-tick marks from the bottom
    ax.set_xticklabels([])  # Remove labels
    ax.tick_params(axis='x', which='both', bottom=False)  # Remove the ticks on the bottom

    plt.tight_layout()

    # Save to PDF if output path is provided
    if output_path:
        with PdfPages(output_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            print(f"Plot saved to {output_path}")

    # Show the plot
    plt.show()
