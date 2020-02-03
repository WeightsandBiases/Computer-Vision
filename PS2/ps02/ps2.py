"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np

import math

# from matplotlib import pyplot as plt


def get_traffic_circle(circles, x_idx, y_len):
    """
    decompose numpy array into a list of traffi_ricle
    Args:
        circles (numpy 3D array): ircles found by cv2.HoughCircles
        x_idx (int): x index to decompose
        y_len(int): length of y
    Returns (tuple): a tuple containing the x, y and radius of a traffic circle
    """
    traffic_circle = list()
    for k in range(y_len):
        traffic_circle.append(circles[0][x_idx][k])
    return tuple(traffic_circle)


def get_traffic_circles(circles):
    """
    find the circles belonging to a traffic sign
    Args: 
        circles (numpy 3D array) circles found by cv2.HoughCircles
    Returns (set): a set of tuples containing the x, y and radius of a 
                   traffic circle
    """
    # int conversion so there are no negative overflows
    circles = circles.astype(int, copy=False)
    # vertically line up three circles of the same size and distance apart
    # indicies
    X_POS_IDX = 0
    R_IDX = 2  # radius
    # tolerane for how different the radii sizes can be
    RADIUS_TOL = 2
    # tolerance for how different the X position can be
    X_POS_TOL = 2
    # store traffic circles
    traffic_circles = set()
    z_len, x_len, y_len = circles.shape
    j_init = 0
    for i in range(x_len):
        j_init += 1
        for j in range(j_init, x_len):
            # check radius and x position difference
            if (
                abs(circles[0][i][R_IDX] - circles[0][j][R_IDX]) < RADIUS_TOL
                and abs(circles[0][i][X_POS_IDX] - circles[0][j][X_POS_IDX]) < X_POS_TOL
            ):
                # decompose np array
                traffic_circles.add(get_traffic_circle(circles, i, y_len))
                traffic_circles.add(get_traffic_circle(circles, j, y_len))
    return traffic_circles


def get_traffic_sign_xy(circles):
    """
    find the x y coordinate of the traffic sign
    Args: 
        circles (numpy 3D array) circles found by cv2.HoughCircles
    Returns (set): a set of tuples containing the x, y and radius of a 
                   traffic circle
    """
    traffic_circles = get_traffic_circles(circles)
    # average out the traffic circles x and y and return the distance
    x_total = 0.0
    y_total = 0.0
    for circle_x, circle_y, circle_r in traffic_circles:
        x_total += circle_x
        y_total += circle_y
    len_circles = len(traffic_circles)
    return (x_total / len_circles, y_total / len_circles)


def get_traffic_state(image, circles):
    """
        Args: 
            image   (numpy.array): image containing a traffic light.
            circles (numpy 3D array): circles found by cv2.HoughCircles
        Returns:
            state (str): string representation of traffic light state
    """
    # Index of y coordinates returned by get_traffic_circles()
    Y_IDX = 1
    traffic_circles = list(get_traffic_circles(circles))
    # sort traffic circles by height
    traffic_circles = sorted(traffic_circles, key=lambda circle: circle[Y_IDX])
    # convert to tuple and unpack
    red_light, yellow_light, green_light = tuple(traffic_circles)
    red_x, red_y, red_r = red_light
    yellow_x, yellow_y, yellow_r = yellow_light
    green_x, green_y, green_r = green_light
    # samples extracted have values in blue, green, red order
    red_samp = image[int(red_y), int(red_x)]
    yellow_samp = image[int(yellow_y), int(yellow_x)]
    green_samp = image[int(green_y), int(green_x)]

    # evaluate lights against thresholds
    RED_THRESH = 200
    YELLOW_THRESH = 200
    GREEN_THRESH = 200
    red_samp_b, red_samp_g, red_samp_r = red_samp
    yellow_samp_b, yellow_samp_g, yellow_samp_r = yellow_samp
    green_samp_b, green_samp_g, green_samp_r = green_samp
    if yellow_samp_g > YELLOW_THRESH and yellow_samp_r > YELLOW_THRESH:
        return "yellow"
    elif red_samp_r > RED_THRESH:
        return "red"
    elif green_samp_g > GREEN_THRESH:
        return "green"
    else:
        raise ValueError


def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    # inverse of the accumulator ratio
    DP = 1
    # the minimum distance between the centers of detected circles
    MIN_DIST = 40
    # higher threshold for histerisis   (lower = more sensitive)
    THRESH_HI = 40
    # the accumulator threshold hold for accumulation to occur
    # (lower = more sensitive)
    ACCUM_THRESH = 8

    # retrieve arguments
    min_radius = min(radii_range)
    max_radius = max(radii_range) + 5

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # use HoughCircles to get the information of detected circles
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        DP,
        MIN_DIST,
        param1=THRESH_HI,
        param2=ACCUM_THRESH,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    traffic_sign_xy = get_traffic_sign_xy(circles)
    traffic_state = get_traffic_state(img_in, circles)

    return (traffic_sign_xy, traffic_state)


def get_ecld_dist(x_1, y_1, x_2, y_2):
    """
    calculates the euclidean distance of two points
    Args:
        x_1, y_1, x_2, y_2(floats): x and y coordinates
    Returns: euclidean distance
    """
    return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def get_len_line(line):
    """
    get length of a line
    Args:
        line (numpy.array): line segment
    Returns:
        (float): length of line
    """
    x_1, y_1, x_2, y_2 = line
    return get_ecld_dist(x_1, y_1, x_2, y_2)


def is_same_len(line_1, line_2, tolerance):
    """
    compare two lines to see if they are the same length
    Args:
        line_1 (numpy.array): line 1 for comparison
        line_2 (numpy.array): line 2 for comparison
        tolerance (float): tolerance for comparison
    Returns:
        True if they are the same length
    """

    return abs(get_len_line(line_1) - get_len_line(line_2)) < tolerance


def avg(args):
    """
    calculate the average
    Args: 
        (iterable)
    Returns:
        (float) average of iterable
    """
    return sum(args) / len(args)


def get_endpts_touch(line_1, line_2, tolerance):
    """ 
    compare two lines to see if the endpoints are touching 
    Args:
        line_1 (numpy.array): line 1 for comparison
        line_2 (numpy.array): line 2 for comparison
        tolerance (float): tolerance for comparison
    Returns:
        endpoints that are touching
    """
    x_1, y_1, x_2, y_2 = line_1
    x_3, y_3, x_4, y_4 = line_2

    # hardcoded distance comparisons, I mean uh, loop unrolled
    # for optimization... yeah...
    if get_ecld_dist(x_1, y_1, x_3, y_3) < tolerance:
        return ("1-3", avg((x_1, x_3)), avg((y_1, y_3)))
    elif get_ecld_dist(x_1, y_1, x_4, y_4) < tolerance:
        return ("1-4", avg((x_1, x_4)), avg((y_1, y_4)))
    elif get_ecld_dist(x_2, y_2, x_3, y_3) < tolerance:
        return ("2-3", avg((x_2, x_3)), avg((y_2, y_3)))
    elif get_ecld_dist(x_2, y_2, x_4, y_4) < tolerance:
        return ("2-4", avg((x_2, x_4)), avg((y_2, y_4)))
    else:
        return None


def is_lines_angle(line_1, line_2, angle, tolerance):
    """
    see if the line segments have a certain angle between them
    Args:
        line_1 (numpy.array): line 1 for comparison
        line_2 (numpy.array): line 2 for comparison
        angle (float): angle to be evaluated in radians
        tolerance (float): tolerance for comparison
    Returns: 
        True if line segments are in an equilateral angle
    """
    x_1, y_1, x_2, y_2 = line_1
    x_3, y_3, x_4, y_4 = line_2
    # compute angle between two points
    alpha_0 = math.atan2(y_2 - y_1, x_2 - x_1)
    alpha_1 = math.atan2(y_4 - y_3, x_4 - x_3)
    theta = alpha_1 - alpha_0
    theta = abs(theta) % (math.pi)
    return abs(theta - angle) < tolerance


def get_centroid(x_coord, y_coord):
    """
    get centroid of x and y points:
    Args:
        x_pts (iterable): x coordinates
        y_pts (iterable): y coordinates
    Return:
        (tuple): x and y centroid of the points
    """
    return (avg(x_coord), avg(y_coord))


def get_eq_triang_xy(line_1, line_2, tolerance):
    """
    get the x and y centroid of an equilateral triangel
    Args:
        line_1 (numpy.array): line 1 for comparison
        line_2 (numpy.array): line 2 for comparison
        tolerance (float): tolerance for touching points
    Returns: 
        (tuple): x and y coordinates of triangle centroid
    """
    x_1, y_1, x_2, y_2 = line_1
    x_3, y_3, x_4, y_4 = line_2
    touch_id, touch_x, touch_y = get_endpts_touch(line_1, line_2, tolerance)
    if touch_id == "1-3":
        return get_centroid((x_2, x_4, touch_x), (y_2, y_4, touch_y))
    elif touch_id == "1-4":
        return get_centroid((x_2, x_3, touch_x), (y_2, y_3, touch_y))
    elif touch_id == "2-3":
        return get_centroid((x_1, x_4, touch_x), (y_1, y_4, touch_y))
    elif touch_id == "2-4":
        return get_centroid((x_1, x_3, touch_x), (y_1, y_3, touch_y))
    else:
        return None


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    # Edge detection parameters
    # sobel aperature
    SOBEL_APT = 3
    # canny high threshold to start the line
    THRESH_HI = 95
    # canny low threshold to continue the line
    THRESH_LO = 70

    # Line detection parameters (line detection uses edge detection)
    # distance resolution of the accumulator in pixels
    RHO = 1
    # angle resolution of the accumulator in pixels
    THETA = np.pi / 180
    # minimum threshold for votes to count in the accumulator
    THRESH_ACCUM = 25
    # minimum length of a line
    MIN_LENGTH = 25
    # maximum gap in between lines
    MAX_GAP = 4

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # find edges using canny
    edges = cv2.Canny(img, THRESH_LO, THRESH_HI, apertureSize=SOBEL_APT)

    # find lines in the image
    lines = cv2.HoughLinesP(
        edges, RHO, THETA, THRESH_ACCUM, minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP
    )

    # --------------- debug
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(cimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("detected circles", cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --------------- debug

    # find a touching point of two line segments of equal length
    # threshold for line segments to be considered touching
    TOUCH_THRESH = 3
    # threshold for length comparison
    LENTH_THRESH = 4
    # angle of equilateral triangle
    ANGLE = math.pi / 3
    # threshold for comparing with equilateral triangle
    # about 5 deg
    THETA_THRESH = math.pi / 36

    lines_dim_z, lines_dim_y, lines_dim_x = lines.shape
    j_init = 0
    for i in range(lines_dim_z):
        j_init += 1
        for j in range(j_init, lines_dim_z):
            line_1 = lines[i][0]
            line_2 = lines[j][0]
            if (
                is_same_len(line_1, line_2, LENTH_THRESH)
                and get_endpts_touch(line_1, line_2, TOUCH_THRESH)
                and is_lines_angle(line_1, line_2, ANGLE, THETA_THRESH)
            ):
                return get_eq_triang_xy(line_1, line_2, TOUCH_THRESH)
    return None


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    # Edge detection parameters
    # sobel aperature
    SOBEL_APT = 3
    # canny high threshold to start the line
    THRESH_HI = 95
    # canny low threshold to continue the line
    THRESH_LO = 70

    # Line detection parameters (line detection uses edge detection)
    # distance resolution of the accumulator in pixels
    RHO = 1
    # angle resolution of the accumulator in pixels
    THETA = np.pi / 180
    # minimum threshold for votes to count in the accumulator
    THRESH_ACCUM = 25
    # minimum length of a line
    MIN_LENGTH = 25
    # maximum gap in between lines
    MAX_GAP = 4

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # find edges using canny
    edges = cv2.Canny(img, THRESH_LO, THRESH_HI, apertureSize=SOBEL_APT)
    # --------------- debug
    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # --------------- debug

    # <==================== find lines in image
    lines = cv2.HoughLinesP(
        edges, RHO, THETA, THRESH_ACCUM, minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP
    )
    # --------------- debug
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(cimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("detected circles", cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --------------- debug

    # <=================== find circles in image that bounds an octagon
    # inverse of the accumulator ratio
    DP = 1
    # the minimum distance between the centers of detected circles
    MIN_DIST = 100
    # higher threshold for histerisis   (lower = more sensitive)
    THRESH_HI = 30
    # the accumulator threshold hold for accumulation to occur
    # (lower = more sensitive)
    ACCUM_THRESH = 8

    # set min and max radius, larger than traffic light circles to filter them
    # out
    MIN_RADIUS = 45
    MAX_RADIUS = 60

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # use HoughCircles to get the information of detected circles
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        DP,
        MIN_DIST,
        param1=THRESH_HI,
        param2=ACCUM_THRESH,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    # --------------- debug
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # circles = np.uint16(np.around(circles))

    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow("detected circles", cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --------------- debug

    # for every circle found check to see if an line exists inside
    # padding for the bounding circle
    BOUND_THRESH = 4
    # minumum number of lines that need to be detected inside the circle
    # for it to qualify as a stop sign
    MIN_LINES = 5
    for circle in circles[0]:
        lines_in_circle = 0
        circle_x, circle_y, circle_r = circle
        x_min = circle_x - circle_r - BOUND_THRESH
        x_max = circle_x + circle_r + BOUND_THRESH
        y_min = circle_y - circle_r - BOUND_THRESH
        y_max = circle_y + circle_r + BOUND_THRESH
        for line in lines:
            x_1, y_1, x_2, y_2 = line[0]
            # check bounds
            if (
                min((x_min, x_1, x_2)) == x_min
                and max((x_max, x_1, x_2)) == x_max
                and min((y_min, y_1, y_2)) == y_min
                and max((y_max, y_1, y_2)) == y_max
            ):
                lines_in_circle += 1
        if lines_in_circle > MIN_LINES:
            return (circle_x, circle_y)
    return None


def get_diamond_xy(line_1, line_2, tolerance):
    """
    get the x and y centroid of a diamond
    Args:
        line_1 (numpy.array): line 1 for comparison
        line_2 (numpy.array): line 2 for comparison
        tolerance (float): tolerance for touching points
    Returns: 
        (tuple): x and y coordinates of diamond centroid
    """
    x_1, y_1, x_2, y_2 = line_1
    x_3, y_3, x_4, y_4 = line_2
    touch_id, touch_x, touch_y = get_endpts_touch(line_1, line_2, tolerance)
    if touch_id == "1-3":
        return get_centroid((x_2, x_4), (y_2, y_4))
    elif touch_id == "1-4":
        return get_centroid((x_2, x_3), (y_2, y_3))
    elif touch_id == "2-3":
        return get_centroid((x_1, x_4), (y_1, y_4))
    elif touch_id == "2-4":
        return get_centroid((x_1, x_3), (y_1, y_3))
    else:
        return None


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # Edge detection parameters
    # sobel aperature
    SOBEL_APT = 3
    # canny high threshold to start the line
    THRESH_HI = 95
    # canny low threshold to continue the line
    THRESH_LO = 70

    # Line detection parameters (line detection uses edge detection)
    # distance resolution of the accumulator in pixels
    RHO = 1
    # angle resolution of the accumulator in pixels
    THETA = np.pi / 180
    # minimum threshold for votes to count in the accumulator
    THRESH_ACCUM = 25
    # minimum length of a line
    MIN_LENGTH = 25
    # maximum gap in between lines
    MAX_GAP = 4

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # find edges using canny
    edges = cv2.Canny(img, THRESH_LO, THRESH_HI, apertureSize=SOBEL_APT)

    # find lines in the image
    lines = cv2.HoughLinesP(
        edges, RHO, THETA, THRESH_ACCUM, minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP
    )

    # find a touching point of two line segments of equal length
    # threshold for line segments to be considered touching
    TOUCH_THRESH = 3
    # threshold for length comparison
    LENTH_THRESH = 4
    # angle of diamond
    ANGLE = math.pi / 2
    # threshold for comparing with equilateral triangle
    # about 5 deg
    THETA_THRESH = math.pi / 36

    # color thresholds
    THRESH_B = 50
    THRESH_G = 200
    THRESH_R = 200

    lines_dim_z, lines_dim_y, lines_dim_x = lines.shape
    j_init = 0
    for i in range(lines_dim_z):
        j_init += 1
        for j in range(j_init, lines_dim_z):
            line_1 = lines[i][0]
            line_2 = lines[j][0]
            if (
                is_same_len(line_1, line_2, LENTH_THRESH)
                and get_endpts_touch(line_1, line_2, TOUCH_THRESH)
                and is_lines_angle(line_1, line_2, ANGLE, THETA_THRESH)
            ):
                dmnd_x, dmnd_y = get_diamond_xy(line_1, line_2, TOUCH_THRESH)
                sample = img_in[int(dmnd_y), int(dmnd_x)]
                sample_b, sample_g, sample_r = sample
                if sample_b < THRESH_B and sample_g > THRESH_G and sample_r > THRESH_R:
                    return (int(dmnd_x), int(dmnd_y))

    return None


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # Edge detection parameters
    # sobel aperature
    SOBEL_APT = 3
    # canny high threshold to start the line
    THRESH_HI = 95
    # canny low threshold to continue the line
    THRESH_LO = 70

    # Line detection parameters (line detection uses edge detection)
    # distance resolution of the accumulator in pixels
    RHO = 1
    # angle resolution of the accumulator in pixels
    THETA = np.pi / 180
    # minimum threshold for votes to count in the accumulator
    THRESH_ACCUM = 25
    # minimum length of a line
    MIN_LENGTH = 25
    # maximum gap in between lines
    MAX_GAP = 4

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # find edges using canny
    edges = cv2.Canny(img, THRESH_LO, THRESH_HI, apertureSize=SOBEL_APT)

    # find lines in the image
    lines = cv2.HoughLinesP(
        edges, RHO, THETA, THRESH_ACCUM, minLineLength=MIN_LENGTH, maxLineGap=MAX_GAP
    )

    # find a touching point of two line segments of equal length
    # threshold for line segments to be considered touching
    TOUCH_THRESH = 3
    # threshold for length comparison
    LENTH_THRESH = 4
    # angle of diamond
    ANGLE = math.pi / 2
    # threshold for comparing with equilateral triangle
    # about 5 deg
    THETA_THRESH = math.pi / 36

    # color thresholds
    THRESH_B = 50
    THRESH_G = 200
    THRESH_R = 200

    lines_dim_z, lines_dim_y, lines_dim_x = lines.shape
    j_init = 0
    for i in range(lines_dim_z):
        j_init += 1
        for j in range(j_init, lines_dim_z):
            line_1 = lines[i][0]
            line_2 = lines[j][0]
            if (
                is_same_len(line_1, line_2, LENTH_THRESH)
                and get_endpts_touch(line_1, line_2, TOUCH_THRESH)
                and is_lines_angle(line_1, line_2, ANGLE, THETA_THRESH)
            ):
                dmnd_x, dmnd_y = get_diamond_xy(line_1, line_2, TOUCH_THRESH)
                sample = img_in[int(dmnd_y), int(dmnd_x)]
                sample_b, sample_g, sample_r = sample
                if sample_b < THRESH_B and sample_g < THRESH_G and sample_r > THRESH_R:
                    return (int(dmnd_x), int(dmnd_y))


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    # inverse of the accumulator ratio
    DP = 1
    # the minimum distance between the centers of detected circles
    MIN_DIST = 50
    # higher threshold for histerisis   (lower = more sensitive)
    THRESH_HI = 40
    # the accumulator threshold hold for accumulation to occur
    # (lower = more sensitive)
    ACCUM_THRESH = 8

    # set min and max radius, larger than traffic light circles to filter them
    # out
    MIN_RADIUS = 35
    MAX_RADIUS = 40

    # convert image to grayscale
    img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)

    # use HoughCircles to get the information of detected circles
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        DP,
        MIN_DIST,
        param1=THRESH_HI,
        param2=ACCUM_THRESH,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    z_len, x_len, y_len = circles.shape

    # indicies
    X_POS_IDX = 0
    Y_POS_IDX = 1
    R_IDX = 2

    # --------------- debug
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # circles = np.uint16(np.around(circles))

    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.imshow("detected circles", cimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --------------- debug

    # evaluate center for white and elswhere for red
    WHITE_THRESH = 200
    RED_THRESH = 200

    for i in range(x_len):
        circle_x = circles[0][i][X_POS_IDX]
        circle_y = circles[0][i][Y_POS_IDX]
        circle_r = circles[0][i][R_IDX]
        samp_white = img_in[int(circle_y), int(circle_x)]
        samp_red = img_in[int(circle_y + circle_r / 2), int(circle_x + circle_r / 2)]
        samp_w_b, samp_w_g, samp_w_r = samp_white
        samp_r_b, samp_r_g, samp_r_r = samp_red
        if (
            samp_w_b > WHITE_THRESH
            and samp_w_g > WHITE_THRESH
            and samp_w_r > WHITE_THRESH
            and samp_r_r > RED_THRESH
        ):
            return (circle_x, circle_y)
    return None


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError


def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    raise NotImplementedError
