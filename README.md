# SVD-DARN

Official implementation of:
"SVD-DARN: Multiscale SVD Profile with attention mechanism for PolSAR Image Classification"
## Dataset
The model was tested with three real PolSAR datasets:

* Flevoland AIRSAR dataset: Fully Polarimetric SAR, image size: 750×1024 pixels, ground truth: 15 land cover classes
* San Francisco AIRSAR dataset: Fully Polarimetric SAR, image size: 900×1024 pixels, ground truth: 5 land cover classes
* San Francisco RADARSAT-2 dataset: Fully Polarimetric SAR, image size: 1380×1800 pixels, ground truth: 5 land cover classes
### Note
This implementation is for DARN block ....Multiscale SVD profile generation done in matlab software....
The SVD-DARN framework operates on MSVDP (Multi-Scale Vector Dual-Polarimetric Profiles) features. 
The MSVDP profile generation was carried out using MATLAB prior to the deep learning stage. 
The generated feature representations were then imported into the Python environment for normalization, patch extraction, and model training.
