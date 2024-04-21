# Towards Viewpoint Robustness for Bird's Eye View Segmentation 

Datasets from CARLA and NVIDIA DRIVE Sim are provided [here](https://drive.google.com/drive/folders/1FQGl9oHyMb7CspUBSFQvpByZD9myLync) for future work in quantifying viewpoint robustness. This repository contains the code for preprocessing and loading this data into your ML pipeline!

> __Towards Viewpoint Robustness in Bird's Eye View Segmentation__  
> [Tzofi Klinghoffer](https://tzofi.github.io/), [Jonah Philion](https://www.cs.toronto.edu/~jphilion/), [Wenzheng Chen](https://www.cs.toronto.edu/~wenzheng/), [Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/), [Jungseock Joo](https://www.jsjoo.com/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Jose M. Alvarez](https://alvarezlopezjosem.github.io/)  
> _International Conference on Computer Vision (_ICCV_), 2023_  
> __[Project page](https://nvlabs.github.io/viewpoint-robustness)&nbsp;/ [Paper](https://nvlabs.github.io/viewpoint-robustness/docs/assets/tzofi2023view.pdf)&nbsp;/ [BibTeX](https://nvlabs.github.io/viewpoint-robustness/docs/assets/tzofi2023view.bib)__

For business inquiries, please submit the [NVIDIA research licensing form](https://www.nvidia.com/en-us/research/inquiries/).

## Preparation
```pip install -r requirements.txt```

## Usage

We provide two datasets, one rendered in CARLA, which includes train and test subsets across 36 different viewpoints, and one rendered in DRIVE Sim, which includes test subsets across 11 viewpoints.

### Using CARLA Data

The provided dataset is rendered from a front-facing camera using NuScenes configuration (specified in nuscenes.json). The dataset includes train and test splits across 36 different viewpoints. Train data is collected in CARLA's Town03 and test data is collected in CARLA's Town05. Each split contains 25k images. To create your dataloader:

1. [ Downloading ] Download the data from the above Google drive link and untar it (tar -xvf filename)
2. [ Data Loading ] Run the following command to instiantiate a dataset:
```
python carla.py path_to_data_eg_pitch0
```

The viewpoint change is described by the folder name. pitch0 is the default view and all other changes in viewpoint are applied to it, e.g. pitch-4 changes the viewpoint from the default view to -4 degrees pitch (all other extrinsic parameters left the same), yaw16 changes the viewpoint from the default view to +16 degrees pitch, height15 changes the viewpoint from the default to +15 inches height, and pitch\_height-8\_12 changes the viewpoint from the default to -8 degrees pitch AND +12 inches height. The default extrinsics and corresponding adjustments are stored in each info.json file.

### Using NVIDIA DRIVE Sim Data

The provided dataset is rendered from a front-facing 120 degree camera with an ftheta lens. In our work, we rectify this data that of a 50 degree pinhole model camera. Once rectified, we can then load the data and use it to evaluate our model. Please refer to the following steps:

1. [ Downloading ] Download the data from the above Google drive link and unzip it
2. [ Optional: data is already rectified ][ Rectifying ] Run the following command to rectify all data (this will take awhile to run):
```
python drivesim.py --dataroot ./DRIVESim_datasets --session 5f8f235e-9304-11ed-8a70-6479f09080c1 --dataset_idx 0 --rectify 1
```
3. [ Optional ][ Visualization ] Run the following command to visualize the images and ground truth: 
```
python drivesim.py --dataroot ./DRIVESim_datasets --session 5f8f235e-9304-11ed-8a70-6479f09080c1 --dataset_idx 0 --vis 1
```
4. [ Data Loading ] Run the following command to instantiate a dataset with the rectified data:
```
python drivesim.py --dataroot ./DRIVESim_datasets --session 5f8f235e-9304-11ed-8a70-6479f09080c1 --dataset_idx 0 --frames rgb_jpeg_rectified
```

Dataset idx refers to which test dataset to load, and corresponds to:
0: default view
2: +1.5 m depth
3: +0.2 m height
4: +0.4 m height
5: +0.6 m height
6: +0.8 m height
7: -5 degrees pitch (looking downwards)
8: -10 degrees pitch (looking downwards)
9: -10 degrees pitch (looking downwards) and +0.6 m height
10: +5 degrees pitch (looking upwards)
11: +10 degrees pitch (looking upwards)

The above visualization step visualizes the unrectified images. The codebase also includes a function for visualizing rectified images to verify the data prior to training.

## Rendering CARLA Data

The code that we use to render the CARLA datasets are provided in [this git issue](https://github.com/NVlabs/viewpoint-robustness/issues/4). While we only rendered RGB images and corresponding 3D bounding box labels in our datasets, depth, segmentation, and other data can also be rendered.

## Thanks

Many thanks to [Alperen Degirmenci](https://scholar.harvard.edu/adegirmenci/home) and [Maying Shen](https://mayings.github.io/). 

This project makes use of a number of awesome projects, including:
* [Lift, Splat, Shoot](https://nv-tlabs.github.io/lift-splat-shoot/)
* [Cross View Transformers](https://github.com/bradyz/cross_view_transformers)
* [Worldsheet](https://worldsheet.github.io/)

Many thanks to the authors of these papers!

## License and Citation

```bibtex
@inproceedings{tzofi2023view,
    author = {Klinghoffer, Tzofi and Philion, Jonah and Chen, Wenzheng and Litany, Or and Gojcic, Zan
        and Joo, Jungseock and Raskar, Ramesh and Fidler, Sanja and Alvarez, Jose M},
    title = {Towards Viewpoint Robustness in Bird's Eye View Segmentation},
    booktitle = {International Conference on Computer Vision},
    year = {2023}
}
```

Copyright © 2023, NVIDIA Corporation. All rights reserved.
