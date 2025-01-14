# Change Log

### ToDos:
    - Testing code OOR
    - Separate Enc/ Dec
    - zarr
    - neuroglancer

### Issues
    - can't take two outputs - out0 and out1 properly
    - need to fix exponential normalization (norm_method = exp w/ exp_trd and exp_ftr)
    - need the ability to take stacks of 2D tif as input (now only 3D tif)
## [0.1.2] - 2025-01-14
    - Refactor code OOR principle
    - Can take stacks of 2D tif as input
    - Separate encoder and decoder operation, latent z saved in hbranch folder
    - Support fp16 while inference
    - Optimize assemble images function
## [0.1.1] - 2024-12-12
    - optimize unnecessary "permute" operation while TTA
    - input_augmentation params using flipX/Y instead of flip2/3
    - add --mc args represent monte carlo over N time 
    - save results path assigned by config yaml
## [0.1.0] - 2024-12-10
    - initial release
### Added
### Changed
### Fixed