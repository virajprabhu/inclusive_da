## PyTorch Code for "Can domain adaptation make object recognition work for everyone?" 
### Learning with Limited and Imperfect Data workshop, CVPR 2022

Table of Contents
=================

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Usage](#usage)
      * [Download data](#data-download)
      * [Train and adapt model](#train-and-adapt-model)
   * [Citation](#citation)

## Setup and Dependencies

1. Create an anaconda environment with atleast [Python 3.6](https://www.python.org/downloads/release/python-365/) and activate: 
```
conda create -n da python=3.6.8
conda activate da
```
2. Navigate into the code directory: ```cd inclusive_da/```
3. Install dependencies: (Takes ~2-3 minutes) 
```
pip install -r requirements.txt
``` 

And you're all set! 

## Usage 

### Download data

To download both Dollarstreet-DA and GeoYFCC-DA datasets, run `sh scripts/download.sh` (note: requires ~12GB of free space).

### Train unsupervised DA model

Run ```python train.py``` to train a source model from scratch followed by unsupervised DA, by passing it appropriate arguments.

Hyperparameter configurations for each benchmark are included as yml files inside the ```config``` folder:

```
python train.py --id <experiment_identifier> \
                --load_from_cfg True \ 
                --cfg_file config/dollarstreet/<da_method>.yml
```

You can pass in in DANN/MMD/SENTRY as `<da_method>`, or implement your own in `adapt/adapt.py`. To run a custom train job, you can either i] Create a new config file (ideally don't edit the existing ones), or ii) manually override the value for certain hyperparameters.

### Citation

If you found our curated dataset or work useful, please consider citing our paper as well as the papers introducing the original data sources: 

```
@inproceedings{prabhu_2022_CVPR,
    author    = {Prabhu, Viraj and Selvaraju, Ramprasaath R. and Hoffman, Judy and Naik, Nikhil},
    title     = {Can Domain Adaptation Make Object Recognition Work for Everyone?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3981-3988}
}

@inproceedings{rojas2022dollar,
  title={The dollar street dataset: Images representing the geographic and socioeconomic diversity of the world},
  author={Rojas, William A Gaviria and Diamos, Sudnya and Kini, Keertan Ranjan and Kanter, David and Reddi, Vijay Janapa and Coleman, Cody},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}

@inproceedings{dubey2021adaptive,
  title={Adaptive methods for real-world domain generalization},
  author={Dubey, Abhimanyu and Ramanathan, Vignesh and Pentland, Alex and Mahajan, Dhruv},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14340--14349},
  year={2021}
}
```