# GHAFDT: ***A Geometric Hippocampus-based Approach for Alzheimer's Disease Cross-species Transfer***

## Requirements and Dependencies
- Dependencies and packages used for this work are available in the `requirements.txt` file. You can create a virtual enviroment and install them in this enviroment. 

## Data and Hippocampus segmentation
- **<ins>Mice</ins>**: We use the approach from Zhou et al., (2023). See procedures [here](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_13)
- **<ins>Human</ins>**: We use the FIRST-FSL toolbox. See procedures [here](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FIRST(2f)StepByStep.html)
- For ADNI and AIBL, the IDs of the subjects used in this work are given in the `subject_ids` folder. Those IDs can be used to download the data from LONI. For OASIS-2, the data can be downloaded from [here](https://sites.wustl.edu/oasisbrains/datasets/) 

## Shape Space Representation
- Run the file `shape_sp_representation.py` in the `shape_sp_representation` folder.
  
## Shape Interpolation
- Run the file `ADNI_interp_shape_space.py` in the `interpolation` folder.

## Validation and Test Split
- Run the file `ADNI_val_test_split.py`in the `val_test_split` folder.

## Karcher Mean and Tangent Projection
- Run the file `ADNI_karcher_and_tan_proj_mice_human.py` in the `karcher_and_tangent_project` folder.

## Domain Alignment and Classification
- Run the file `ADNI_mice_align_and_classify.py` in the `domain_alignment_and_classification` folder.

## Other Compared methods
- Files can be found under the `comaprison_files` folder.


