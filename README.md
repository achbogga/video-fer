# video-fer
Video Facial Expression Recognition (MS thesis)

## Getting Started - Prerequisites
* Keras, Tensorflow, opencv-python, ffmpeg, imageio
* Highly recommend using a GPU for this project

### Installing on a native linux machine

#### Installing Tensorflow and other python dependencies
* Install NVIDIA CUDA Toolkit. Most stable cuda, cudnn versions are CUDA 8.0 and CUDNN 6
* Highly recommended to use virtualenv while executing the pip installation step so that native python environment is not messed up and easy to maintain in future.
* Install the tensorflow library from the sources for python 2.7.x as per the instructions given in the [https://www.tensorflow.org/install/install_sources](https://www.tensorflow.org/install/install_sources).
* Install other python dependencies using pip (in the same virtualenv created)
* Change the working directory to the source repository and execute the following command. ```cd video-fer && pip install -yUr requirements.txt```

## Running the core VIDEO-FER Pipeline (Test installation)

Activate the virtualenv in which the project is installed into.

Change into working directory video-fer ```cd video-fer```

### Structuring your data
* All the data should be placed in the Data folder under video-fer/Data.

###Preprocessing: MTCNN Alignment of the data 
## Built With

* [Tensorflow](https://www.tensorflow.org/) - The deep learning python framework used
* [OpenCV](https://opencv.org) - The Computer Vision Library used for image processing. 

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* [**Achyut Boggaram**](mailto:achbogga@umail.iu.edu)

## Acknowledgments

* Prof. David Crandall
* IUB SICE CS Dept.
