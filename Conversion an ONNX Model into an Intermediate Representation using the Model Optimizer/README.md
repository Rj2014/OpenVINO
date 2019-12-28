First download the bvlc_alexnet model using 
`wget https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_alexnet.tar.gz` 

Then run the following command to unpack it 
`tar -xvf bvlc_alexnet.tar.gz` 

Inside **bvlc_alexnet** directory run 
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model.onnx` to generate respective **.bin** and **.xml** file.
