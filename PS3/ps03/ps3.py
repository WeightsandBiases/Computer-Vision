"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import math


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    x_1, y_1 = p0
    x_2, y_2 = p1

    return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """

    raise NotImplementedError


def sort_by_return(list_of_tuples):
    """
    sort list of tuples for a specific return template
    Args:
        list_of_tuples (list): list of 4 x,y tuples
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    list_of_tuples = sorted(list_of_tuples, key=lambda item: item[0])
    left_side = list_of_tuples[0:2]
    right_side = list_of_tuples[2:4]
    left_side = sorted(left_side, key=lambda item: item[1])
    right_side = sorted(right_side, key=lambda item: item[1])
    result = left_side + right_side
    return result


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    # Tolerance for threshold
    TOL = 0.02
    # number of markers
    N_MARKERS = 4
    # -----------------De Noise Image ------------------------------
    # create copies as to not distrub the original images
    img = np.copy(image)
    templ = np.copy(template)
    # denoising parameters
    # from the openCv docs
    # Parameter regulating filter strength for luminance component.
    # Bigger h value perfectly removes noise but also removes image details,
    # smaller h value preserves details but also preserves some noise
    H = 25
    # The same as h but for color components.
    # For most images value equals 10 will be enough to remove colored noise
    # and do not distort colors
    H_COLOR = 15
    # Size in pixels of the template patch that is used to compute weights.
    # Should be odd. Recommended value 7 pixels
    TEMPLATE_WINDOW_SIZE = 7
    # Size in pixels of the window that is used to compute weighted average for
    # given pixel. Should be odd. Affect performance linearly:
    # greater searchWindowsSize - greater denoising time.
    # Recommended value 21 pixels
    SEARCH_WINDOW_SIZE = 21

    img = cv2.fastNlMeansDenoisingColored(
        img, None, H, H_COLOR, TEMPLATE_WINDOW_SIZE, SEARCH_WINDOW_SIZE
    )
    # -----------------End De Noise Image ------------------------------

    # convert to greyscale and floating point representation
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    templ_grey = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    img_grey = img_grey.astype(np.float32, copy=False)
    templ_grey = templ_grey.astype(np.float32, copy=False)
    # find matches of the templates
    match = cv2.matchTemplate(img_grey, templ_grey, cv2.TM_CCOEFF_NORMED)
    # template centerpoint threshold
    threshold = np.amax(match)
    threshold -= TOL
    # filter points by threshold
    loc = np.argwhere(match >= threshold)
    # obtain points in a list of tuples
    loc = loc.astype(np.int16, copy=False)
    # translate the templated return back to original image
    templ_height, templ_width = templ_grey.shape
    templ_x_offset = templ_width // 2
    templ_y_offset = templ_height // 2
    result = list(zip(loc.T[1] + templ_x_offset, loc.T[0] + templ_y_offset))
    # handle corner cases
    if len(result) > 2 * N_MARKERS:
        del result
        result = list()
        # deal with edge cases of real image
        # inverse of the accumulator ratio
        DP = 1
        # the minimum distance between the centers of detected circles
        MIN_DIST = 300
        # higher threshold for histerisis   (lower = more sensitive)
        THRESH_HI = 40
        # the accumulator threshold hold for accumulation to occur
        # (lower = more sensitive)
        ACCUM_THRESH = 20

        # set min and max radius, larger than traffic light circles to filter them
        # out
        MIN_RADIUS = 30
        MAX_RADIUS = 42

        # convert image to grayscale
        img_grey = img_grey.astype(np.uint8, copy=False)
        # use HoughCircles to get the information of detected circles
        circles = cv2.HoughCircles(
            img_grey,
            cv2.HOUGH_GRADIENT,
            DP,
            MIN_DIST,
            param1=THRESH_HI,
            param2=ACCUM_THRESH,
            minRadius=MIN_RADIUS,
            maxRadius=MAX_RADIUS,
        )

        # --------------- debug
        # circles = np.uint16(np.around(circles))

        # for i in circles[0, :]:
        #     # draw the outer circle
        #     cv2.circle(img_grey, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #     # draw the center of the circle
        #     cv2.circle(img_grey, (i[0], i[1]), 2, (0, 0, 255), 3)
        # cv2.imshow("detected circles", img_grey)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # --------------- debug

        z_len, x_len, y_len = circles.shape
        # indicies
        X_POS_IDX = 0
        Y_POS_IDX = 1
        for i in range(x_len):
            circle_x = circles[0][i][X_POS_IDX]
            circle_y = circles[0][i][Y_POS_IDX]
            result.append((int(circle_x), int(circle_y)))
    # sort the results
    sorted_result = sort_by_return(result)
    return sorted_result


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    COLOR = (0, 255, 0)
    image_out = np.copy(image)
    tuple(markers)
    p1, p2, p3, p4 = markers
    cv2.line(image_out, p1, p2, COLOR, thickness=thickness)
    cv2.line(image_out, p2, p3, COLOR, thickness=thickness)
    cv2.line(image_out, p3, p4, COLOR, thickness=thickness)
    cv2.line(image_out, p4, p1, COLOR, thickness=thickness)
    return image_out


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """

    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = None

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    raise NotImplementedError
