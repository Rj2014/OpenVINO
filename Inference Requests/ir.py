import argparse
import cv2
import time
from helpers import load_to_IE, preprocessing

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():

    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    m_desc = "The location of the model XML file"
    i_desc = "The location of the image input"
    r_desc = "The type of inference request: Async ('A') or Sync ('S')"
    parser.add_argument("-m", help=m_desc)
    parser.add_argument("-i", help=i_desc)
    parser.add_argument("-r", help=i_desc)
    args = parser.parse_args()
    return args


def async_inference(exec_net, input_blob, image):
    exec_net.start_async(request_id=0, inputs={input_blob: image})
    while True:
        status = exec_net.requests[0].wait(-1)
        if status == 0:
            break
        else:
            time.sleep(1)
    return exec_net


def sync_inference(exec_net, input_blob, image):
    result = exec_net.infer({input_blob: image})

    return result


def perform_inference(exec_net, request_type, input_image, input_shape):
    image = cv2.imread(input_image)
    n, c, h, w = input_shape
    preprocessed_image = preprocessing(image, h, w)
    input_blob = next(iter(exec_net.inputs))
    request_type = request_type.lower()
    if request_type == 'a':
        output = async_inference(exec_net, input_blob, preprocessed_image)
    elif request_type == 's':
        output = sync_inference(exec_net, input_blob, preprocessed_image)
    else:
        print("Unknown inference request type, should be 'A' or 'S'.")
        exit(1)

    return output


def main():
    args = get_args()
    exec_net, input_shape = load_to_IE(args.m, CPU_EXTENSION)
    perform_inference(exec_net, args.r, args.i, input_shape)


if __name__ == "__main__":
    main()
