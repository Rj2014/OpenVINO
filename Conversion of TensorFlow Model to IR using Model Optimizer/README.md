First download the SSD MobileNet V2 COCO model from 
`http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

Use the command with the downloaded file to unpack it.
`tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`

Then goto the following Directory
`cd ssd_mobilenet_v2_coco_2018_03_29`
You will see following files 
checkpoint, model.ckpt.index, saved_model, frozen_inference_graph.pb, model.ckpt.meta, model.ckpt.data-00000-of-00001, pipeline.config.

To convert the SSD MobileNet V2 model from TensorFlow:
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
`
Where
`python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py` is a model optimizer file in OpenVINO.

`--input_model frozen_inference_graph.pb` is an input model which is frozen inference graph.

`--tensorflow_object_detection_api_pipeline_config pipeline.config` we feed in the pipeline config file for the Object Detection Model Zoo items

`--reverse_input_channels ` we reverse the Input Channels as they are RGB and we want BGR

After all this, we will have frozen_inference_graph.bin and frozen_inference_graph.xml file to be used with the Inference Engine.
