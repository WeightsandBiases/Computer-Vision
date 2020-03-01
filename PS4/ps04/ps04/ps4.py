"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    img = np.copy(image)
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=0.125)


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    img = np.copy(image)
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=0.125)


def getUniformKernel(k_size):
    """
    Can't believe cv2 doesn't have this built in.
    It generates a uniform kernel.
    Args: 
        k_size (int): size of averaging kernel to use for weighted
                averages. Here we assume the kernel window is a
                square so you will use the same value for both
                width and height.
    Returns:
        (numpy.array): uniform kernel
    """
    # make a 2D square array of 1s
    dim = (k_size, k_size)
    return np.ones(dim)


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """
    kernel = None
    USE_SRC_DEPTH = -1
    THRESH = 1e-6
    # copy and convert images to to 64 bit float
    img_a = np.copy(img_a).astype(np.float64)
    img_b = np.copy(img_b).astype(np.float64)
    if k_type == 'uniform':
        kernel = getUniformKernel(k_size)
    elif k_type == 'gaussian':
        kernel = cv2.getGaussianKernel(k_size, sigma, ktype=cv2.CV_64F)
    # compute gradients
    I_t = img_a - img_b
    I_x = gradient_x(img_a)
    I_y = gradient_y(img_a)

    # compute sum squared differences
    # Sum of the squares of the difference between each x and the mean x value.
    S_xx = cv2.filter2D(I_x ** 2, USE_SRC_DEPTH, kernel)
    # Sum of the squares of the difference between each y and the mean y value.
    S_yy = cv2.filter2D(I_y ** 2, USE_SRC_DEPTH, kernel)
    # Sum of the squares of the difference between each x and the mean y value.
    S_xy = cv2.filter2D(I_x * I_y, USE_SRC_DEPTH, kernel)
    # Sum of squared differences with It
    S_xt = cv2.filter2D(I_x * I_t, USE_SRC_DEPTH, kernel)
    S_yt = cv2.filter2D(I_y * I_t, USE_SRC_DEPTH, kernel)

    # compute determinate with threshold
    M_det = np.clip(S_xx * S_yy - S_xy ** 2, THRESH, np.inf).astype(np.float64)
    # find U V using sum squared differences
    U = -(S_yy * (-S_xt) + (-S_xy) * (-S_yt))
    V = -(S_xx * (-S_yt) + (-S_xy) * (-S_xt))
    # eigen values should not be computed if they are too small
    # avoid division by zero
    U = np.where(M_det != 0.0, U / M_det, 0.0)
    V = np.where(M_det != 0.0, V / M_det, 0.0)
    return (U, V)


def get_pyramid_kernel(k_type="reduce"):
    """
    Creates a kernel used by reduce_image and expand_image for pyramid
    building.

    Args:
        type (str): the type kernel used ("reduce" or "expand")
    """
    # From Piazza
    # For reduce image, the kernel is [1, 4, 6, 4, 1] / 16
    if k_type == "reduce":
        k_def = np.array([0.0625,  0.25,    0.375,   0.25, 0.0625])
    # For expand image, the kernel is [1, 4, 6, 4, 1] / 8
    elif k_type == "expand":
        k_def = np.array([0.125,  0.5, 0.75,  0.5, 0.125])
    else:
        raise KeyError
    
    return np.outer(k_def, k_def)


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    USE_SRC_DEPTH = -1
    img = np.copy(image)
    kernel = get_pyramid_kernel(k_type="reduce")
    # convolve with filter
    filtered_image = cv2.filter2D(img, USE_SRC_DEPTH, kernel)
    # down sample every other pixel
    return filtered_image[::2, ::2]


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid = list()
    # The first element in the list ([0]) should contain the input image
    img = np.copy(image)
    pyramid.append(img)

    # All other levels contain a reduced version of the previous level
    for i in range(1, levels):
        img = reduce_image(img)
        pyramid.append(img)
    return pyramid


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    # get dimensions of the first image
    combined_img = normalize_and_scale(img_list[0])
    first_height, first_width = img_list[0].shape[:2]
    for img in img_list:
        img_norm = normalize_and_scale(img)
        img_height, img_width = img.shape[:2]
        # fill borders
        if first_height != img_height:
            border = np.zeros((first_height - img_height, img_width))
            img_resized = np.vstack((img_norm, border))
            combined_img = np.hstack((combined_img, img_resized))
    return combined_img


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """

    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """

    raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """

    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    raise NotImplementedError
