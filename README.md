# DeepLab_Caffe2
[Googles DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) Semantic Segmentation ported from TensorFlow to Caffe2

As in TF, the input is transformed as (img = img/127 - 1) to bring in range [-1, 1].
Output is 21 category semeantic lables as in [Pascal VOC](https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt)

# note
This implementation uses an additional operator for managing padding, which can be found in 'customOP' folder. Library for the same is provided in 'OpLibs_CPU'.
currently only CPU version of OP is implemented.


![Alt text](SampleOutput/img.jpg "Input")
