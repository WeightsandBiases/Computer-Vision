import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    red = image[:, :, 2].copy()     # a copy of the red channel of the image
    return red


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    green = image[:, :, 1].copy()   # a copy of the green channel of the image
    return green


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    blue = image[:, :, 0].copy()    # a copy of the blue channel of the image
    return blue


def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    # extract copies
    temp_image = np.copy(image)
    green = temp_image[:, :, 1].copy()
    blue = temp_image[:, :, 0].copy()
    # swap colors
    temp_image[:, :, 0] = green  # blue is now green
    temp_image[:, :, 1] = blue   # green is now blue

    return temp_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    height, width = shape
    src_height, src_width = src.shape
    dst_height, dst_width = dst.shape
    # calculate the range for the middle of the image
    src_height_min = int(src_height / 2 - height / 2)
    src_height_max = int(src_height / 2 + height / 2)
    src_width_min = int(src_width / 2 - width / 2)
    src_width_max = int(src_width / 2 + width / 2)
    dst_height_min = int(dst_height / 2 - height / 2)
    dst_height_max = int(dst_height / 2 + height / 2)
    dst_width_min = int(dst_width / 2 - width / 2)
    dst_width_max = int(dst_width / 2 + width / 2)
    # crop the middle of source image
    src_cropped = src[src_height_min:src_height_max,
                      src_width_min:src_width_max].copy()
    # insert cropped source image into a copy of destination
    # image
    dst_copy = np.copy(dst)
    dst_copy[dst_height_min:dst_height_max,
             dst_width_min:dst_width_max] = src_cropped
    return dst_copy


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    min = np.amin(image).astype(np.float)
    max = np.amax(image).astype(np.float)
    mean = np.mean(image).astype(np.float)
    stddev = np.std(image).astype(np.float)
    return (min, max, mean, stddev)


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    mean = np.mean(image)
    stddev = np.std(image)
    res = (image - mean) / stddev * scale + mean
    return res


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.
 
    Returns:
        numpy.array: Output shifted 2D image.
    """
    height, width = image.shape
    # matrix for shifting
    M = np.float32([[1, 0, -shift],
                    [0, 1, 0]])
    res = cv2.warpAffine(np.copy(image).astype(float),
                         M,
                         (width, height),
                         borderMode=cv2.BORDER_REPLICATE)
    return res


def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    res = np.copy(img1).astype(np.float) - np.copy(img2).astype(np.float)
    cv2.normalize(res, res, 0, 255, cv2.NORM_MINMAX)
    return res


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    res = np.copy(image).astype(np.float)
    height, width, nchannels = image.shape
    # add gaussian noise to the selected channel
    selected_channel = res[:, :, channel].copy()
    noise = np.copy(selected_channel)
    cv2.randn(noise, 0, sigma)
    selected_channel = cv2.add(selected_channel, noise)
    res[:, :, channel] = selected_channel
    return res
