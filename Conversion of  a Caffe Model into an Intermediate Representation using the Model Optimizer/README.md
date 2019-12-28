First download the SqueezeNet V1.1 model by cloning the repository SqueezeNet V1.1 using command
`git clone https://github.com/DeepScale/SqueezeNet`

Then GOTO **SqueezeNet** directory and then to **SqueezeNet_v1.1** directory.

To convert the Squeezenet V1.1 model from Caffe:
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
`

You will see the **.bin** and **.xml** file in the directory.
