# DeepLab_Caffe2

[DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) is Deep Semantic Segmentation project from Google. It currently supports two networks (MobileNetV2 and Xception), and are implementeds in TensorFlow.

In this project, [Googles DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) Semantic Segmentation (MobileNet version) is ported from TensorFlow to Caffe2(which is more embeded friendly).  


As in TF, the input is transformed as (img = img/127 - 1) to bring in range [-1, 1]. 
Output is 21 category semeantic lables as in [Pascal VOC](https://github.com/NVIDIA/DIGITS/blob/master/examples/semantic-segmentation/pascal-voc-classes.txt).  
Input resolution : 513, 513.  
Output resolution: 65, 65.  



# note
This implementation uses an additional operator for managing padding, which can be found in 'customOP' folder. Library for the same is provided in 'OpLibs_CPU'. 
Currently only CPU version of OP is implemented.

# sample result
<p align="center">
  <img src="SampleOutput/img.jpg" width="200" />
  <img src="SampleOutput/visualize.jpg" width="200"/>
  <img src="SampleOutput/lable.jpg" width="200"/>
</p>
-This Mobile Net version is good for implementation on mobile devices with model size around 10MB.  

-Xception Net version (not currently ported) gives better accuracy at cost of model size and processing time.  
