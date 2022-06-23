import cv2
import os
import glob

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



def getVideoFilesFromFolder(dirPath):
    types = (
    dirPath + os.sep + '*.webm', dirPath + os.sep + '*.avi', dirPath + os.sep + '*.mkv', dirPath + os.sep + '*.mp4',
    dirPath + os.sep + '*.mp3', dirPath + os.sep + '*.flac', dirPath + os.sep + '*.ogg')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))
    return files_grabbed
