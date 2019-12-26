## `app.py`

Within `app.py`, the main work is just to call `preprocess_input` and `handle_output` in
the correct locations. You can feed `args.m` into these so they receive the model type,
and will return the appropriate preprocessing or output handling function. You can then feed
the input image or output in as applicable.

The rest of the app will then create the relevant output images so you can see the Inference
Engine at work with the Pre-Trained Models.

Here are the commands used to run the app for each:

```
python app.py -i "images/blue-car.jpg" -t "CAR_META" -m "/home/workspace/models/vehicle-attributes-recognition-barrier-0039.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
```

```
python app.py -i "images/sign.jpg" -t "TEXT" -m "/home/workspace/models/text-detection-0004.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
```

```
python app.py -i "images/sitting-on-car.jpg" -t "POSE" -m "/home/workspace/models/human-pose-estimation-0001.xml" -c "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
```
