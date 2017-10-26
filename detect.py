import os
import sys
import cv2
import caffe
import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import scipy

# Disable most output from Caffe
os.environ['GLOG_minloglevel'] = '2'


def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer

    Arguments:
        deploy_file -- path to a .prototxt file

    Keyword arguments:
        mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t


def resize_image(image, height, width):
    """
    Load a single image

    Returns an np.ndarray (channels x width x height)

    Arguments:
        image_path -- path to image
        width -- resize dimension
        height -- resize dimension

    Keyword arguments:
        mode -- the PIL mode that the image should be converted to
            (RGB for color or L for grayscale)
    """
    return scipy.misc.imresize(image, (height, width), 'bilinear')


def forward_pass(image, net, transformer):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
        images -- a list of np.ndarrays
        net -- a caffe.Net
        transformer -- a caffe.io.Transformer

    Keyword arguments:
        batch_size -- how many images can be processed at once
            (a high value may result in out-of-memory errors)
    """

    image_data = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = image_data
    output = net.forward()[net.outputs[-1]]
    return output


if __name__ == '__main__':

    # Caffe required files
    caffemodel = 'snapshot_iter_70800.caffemodel'
    deploy_file = 'deploy.prototxt'
    mean_file = None

    # Setup network
    caffe.set_mode_gpu()
    net = caffe.Net(deploy_file, caffemodel, caffe.TEST)
    transformer = get_transformer(deploy_file, mean_file=None)
    _, channels, height, width = transformer.inputs['data']

    # Read user input
    video_source = sys.argv[1]
    # Not checking if video_source is correct
    video_capture = cv2.VideoCapture(video_source)

    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out_video = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        image = resize_image(frame, height, width)
        scores = forward_pass(image, net, transformer)
        if scores is None:
            print("No detections found.")
        else:
            for output in scores:
                for left, top, right, bottom, confidence in output:
                    if confidence == 0:
                        continue

                    x1 = int(round(left))
                    y1 = int(round(top))
                    x2 = int(round(right))
                    y2 = int(round(bottom))
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1, lineType=cv2.CV_AA)
            out_video.write(image)
            cv2.imshow('Video Input', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    video_capture.release()
    out_video.release()



























