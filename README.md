# Identity documents detection

The goal of this project course is to approach the problem of automatically identify identity documents (a full report 
is available [here](https://drive.google.com/file/d/1j2SO4XfwxPW3ba3Wy0RxpuncDR3cl0tF/view?usp=sharing)). 
All the work was made possible by the [MIDV-500](https://arxiv.org/abs/1807.05786) dataset.

### Dependencies

The software was developed using:
* Python 3.6
* opencv-contrib-pyhton 3.4.2.16
* opencv-python 3.4.2.16

## Running the code

The two scripts, in order to be executed, need of the MIDV-500 dataset.

### Histogram.py

This script, for each document, selects the best photos inside each folder. The PATH variable, inside the code, needs 
to be set to the local path where the MIDV-500 dataset is located. Then the code can be executed with:

```
python3 Histogram.py
```

### DocExtractor.py

This script tries to extract an identity document from an image. To obtain the best results, models of the identity 
documents need to be created (see the report). The code can be executed with (the last parameter is optional):

```
python3 DocExtractor.py path_to_the_query_img path_to_the_train_img True

```
where the 'query image' should be the model and the 'train image' should be the image that contains within it the id 
document that we want to extract. The last parameter, if set to 'True', allows to output additional information about 
the mapping.