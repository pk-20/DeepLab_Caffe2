# DeepLab_Caffe2
[Googles DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) Semantic Segmentation ([Mobile Net version]()) ported from TensorFlow to Caffe2.  


As in TF, the input is transformed as (img = img/127 - 1) to bring in range [-1, 1]. 
Output is 21 category semeantic lables as in [Pascal VOC](https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt).  
Input Dim: 513, 513.  
Output Dims: 65, 65.  



# note
This implementation uses an additional operator for managing padding, which can be found in 'customOP' folder. Library for the same is provided in 'OpLibs_CPU'. 
Currently only CPU version of OP is implemented.

# sample result
<p align="center">
  <img src="SampleOutput/img.jpg" width="200" />
  <img src="SampleOutput/visualize.jpg" width="200"/>
  <img src="SampleOutput/lable.jpg" width="200"/>
</p>
