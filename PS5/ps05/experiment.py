"""Problem Set 5: Object Tracking and Pedestrian Detection"""

import cv2
import ps5
import os
import numpy as np

# I/O directories
input_dir = "input_images"
output_dir = "output"

NOISE_1 = {"x": 2.5, "y": 2.5}
NOISE_2 = {"x": 7.5, "y": 7.5}


# Helper code


def run_particle_filter_5(
    filter_class,
    imgs_dir,
    template_rect1,
    template_rect2,
    template_rect3,
    save_frames={},
    **kwargs
):
    """Runs a particle filter on a given video and template.
    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.
    Do not modify this function except for the debugging flag.
    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.
    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir) if f[0] != "." and f.endswith(".jpg")]
    imgs_list.sort()

    # Initialize objects
    template1 = None
    pf1 = None
    template2 = None
    pf2 = None
    template3 = None
    pf3 = None
    # frames of important scenes
    template_1_end = 60
    template_3_start = 28
    template_2_end = 47
    frame_num = 0

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if frame_num <= template_1_end:
            if template1 is None:
                template1 = frame[
                    int(template_rect1["y"]) : int(
                        template_rect1["y"] + template_rect1["h"]
                    ),
                    int(template_rect1["x"]) : int(
                        template_rect1["x"] + template_rect1["w"]
                    ),
                ]
                cv2.imwrite("template_1.png", template1)
                if "template" in save_frames:
                    cv2.imwrite(save_frames["template"], template)

                pf1 = filter_class(frame, template1, **kwargs)

            # Process frame
            pf1.process(frame)

        if frame_num <= template_2_end:
            if template2 is None:
                template2 = frame[
                    int(template_rect2["y"]) : int(
                        template_rect2["y"] + template_rect2["h"]
                    ),
                    int(template_rect2["x"]) : int(
                        template_rect2["x"] + template_rect2["w"]
                    ),
                ]
                cv2.imwrite("template_2.png", template2)
                if "template" in save_frames:
                    cv2.imwrite(save_frames["template"], template)

                pf2 = filter_class(frame, template2, **kwargs)

            # Process frame
            pf2.process(frame)

        if frame_num >= template_3_start:
            if template3 is None:
                template3 = frame[
                    int(template_rect3["y"]) : int(
                        template_rect3["y"] + template_rect3["h"]
                    ),
                    int(template_rect3["x"]) : int(
                        template_rect3["x"] + template_rect3["w"]
                    ),
                ]
                cv2.imwrite("template_3.png", template3)
                if "template" in save_frames:
                    cv2.imwrite(save_frames["template"], template)

                pf3 = filter_class(frame, template3, **kwargs)

            # Process frame
            pf3.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            if frame_num <= template_1_end:
                pf1.render(out_frame)
            if frame_num <= template_2_end:
                pf2.render(out_frame)
            if frame_num >= template_3_start:
                pf3.render(out_frame)
            cv2.imshow("Tracking", out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            if frame_num <= template_1_end:
                pf1.render(frame_out)
            if frame_num <= template_2_end:
                pf2.render(frame_out)
            if frame_num >= template_3_start:
                pf3.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print("Working on frame {}".format(frame_num))


def run_particle_filter(
    filter_class, imgs_dir, template_rect, save_frames={}, **kwargs
):
    """Runs a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any
    keyword arguments.

    Do not modify this function except for the debugging flag.

    Args:
        filter_class (object): particle filter class to instantiate
                           (e.g. ParticleFilter).
        imgs_dir (str): path to input images.
        template_rect (dict): template bounds (x, y, w, h), as float
                              or int.
        save_frames (dict): frames to save
                            {<frame number>|'template': <filename>}.
        **kwargs: arbitrary keyword arguments passed on to particle
                  filter class.

    Returns:
        None.
    """

    imgs_list = [f for f in os.listdir(imgs_dir) if f[0] != "." and f.endswith(".jpg")]
    imgs_list.sort()

    # Initialize objects
    template = None
    pf = None
    frame_num = 0
    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template is None:
            template = frame[
                int(template_rect["y"]) : int(template_rect["y"] + template_rect["h"]),
                int(template_rect["x"]) : int(template_rect["x"] + template_rect["w"]),
            ]

            if "template" in save_frames:
                cv2.imwrite(save_frames["template"], template)

            pf = filter_class(frame, template, **kwargs)

        # Process frame
        pf.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf.render(out_frame)
            cv2.imshow("Tracking", out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            pf.render(frame_out)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print("Working on frame {}".format(frame_num))


def run_kalman_filter(kf, imgs_dir, noise, sensor, save_frames={}, template_loc=None):

    imgs_list = [f for f in os.listdir(imgs_dir) if f[0] != "." and f.endswith(".jpg")]
    imgs_list.sort()

    frame_num = 0

    if sensor == "hog":
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    elif sensor == "matching":
        frame = cv2.imread(os.path.join(imgs_dir, imgs_list[0]))
        template = frame[
            template_loc["y"] : template_loc["y"] + template_loc["h"],
            template_loc["x"] : template_loc["x"] + template_loc["w"],
        ]

    else:
        raise ValueError("Unknown sensor name. Choose between 'hog' or " "'matching'")

    for img in imgs_list:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Sensor
        if sensor == "hog":
            (rects, weights) = hog.detectMultiScale(
                frame, winStride=(4, 4), padding=(8, 8), scale=1.05
            )

            if len(weights) > 0:
                max_w_id = np.argmax(weights)
                z_x, z_y, z_w, z_h = rects[max_w_id]

                z_x += z_w // 2
                z_y += z_h // 2

                z_x += np.random.normal(0, noise["x"])
                z_y += np.random.normal(0, noise["y"])

        elif sensor == "matching":
            corr_map = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF)
            z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)

            z_w = template_loc["w"]
            z_h = template_loc["h"]

            z_x += z_w // 2 + np.random.normal(0, noise["x"])
            z_y += z_h // 2 + np.random.normal(0, noise["y"])

        x, y = kf.process(z_x, z_y)

        if False:  # For debugging, it displays every frame
            out_frame = frame.copy()
            cv2.circle(out_frame, (int(z_x), int(z_y)), 20, (0, 0, 255), 2)
            cv2.circle(out_frame, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.rectangle(
                out_frame,
                (int(z_x) - z_w // 2, int(z_y) - z_h // 2),
                (int(z_x) + z_w // 2, int(z_y) + z_h // 2),
                (0, 0, 255),
                2,
            )

            cv2.imshow("Tracking", out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num in save_frames:
            frame_out = frame.copy()
            cv2.circle(frame_out, (int(x), int(y)), 10, (255, 0, 0), 2)
            cv2.imwrite(save_frames[frame_num], frame_out)

        # Update frame number
        frame_num += 1
        if frame_num % 20 == 0:
            print("Working on frame {}".format(frame_num))


def part_1b():
    print("Part 1b")

    template_loc = {"y": 72, "x": 140, "w": 50, "h": 50}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = 0.2 * np.eye(4)  # Process noise array
    R = 0.5 * np.eye(4)  # Measurement noise array

    kf = ps5.KalmanFilter(template_loc["x"], template_loc["y"])

    save_frames = {
        10: os.path.join(output_dir, "ps5-1-b-1.png"),
        30: os.path.join(output_dir, "ps5-1-b-2.png"),
        59: os.path.join(output_dir, "ps5-1-b-3.png"),
        99: os.path.join(output_dir, "ps5-1-b-4.png"),
    }

    run_kalman_filter(
        kf,
        os.path.join(input_dir, "circle"),
        NOISE_2,
        "matching",
        save_frames,
        template_loc,
    )


def part_1c():
    print("Part 1c")

    init_pos = {"x": 311, "y": 217}

    # Define process and measurement arrays if you want to use other than the
    # default. Pass them to KalmanFilter.
    Q = 0.2 * np.eye(4)  # Process noise array
    R = 0.5 * np.eye(4)  # Measurement noise array

    kf = ps5.KalmanFilter(init_pos["x"], init_pos["y"])

    save_frames = {
        10: os.path.join(output_dir, "ps5-1-c-1.png"),
        33: os.path.join(output_dir, "ps5-1-c-2.png"),
        84: os.path.join(output_dir, "ps5-1-c-3.png"),
        159: os.path.join(output_dir, "ps5-1-c-4.png"),
    }

    run_kalman_filter(
        kf, os.path.join(input_dir, "walking"), NOISE_1, "hog", save_frames
    )


def part_2a():

    template_loc = {"y": 72, "x": 140, "w": 50, "h": 50}

    save_frames = {
        10: os.path.join(output_dir, "ps5-2-a-1.png"),
        30: os.path.join(output_dir, "ps5-2-a-2.png"),
        59: os.path.join(output_dir, "ps5-2-a-3.png"),
        99: os.path.join(output_dir, "ps5-2-a-4.png"),
    }

    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    run_particle_filter(
        ps5.ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "circle"),
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc,
    )  # Add more if you need to


def part_2b():

    template_loc = {"x": 360, "y": 141, "w": 127, "h": 179}

    save_frames = {
        10: os.path.join(output_dir, "ps5-2-b-1.png"),
        33: os.path.join(output_dir, "ps5-2-b-2.png"),
        84: os.path.join(output_dir, "ps5-2-b-3.png"),
        99: os.path.join(output_dir, "ps5-2-b-4.png"),
    }
    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)
    run_particle_filter(
        ps5.ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate_noisy"),
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc,
    )  # Add more if you need to


def part_3():
    template_rect = {"x": 538, "y": 377, "w": 73, "h": 117}
    save_frames = {
        22: os.path.join(output_dir, "ps5-3-a-1.png"),
        50: os.path.join(output_dir, "ps5-3-a-2.png"),
        160: os.path.join(output_dir, "ps5-3-a-3.png"),
    }

    num_particles = 1000  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 12  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.3  # Set a value for alpha

    run_particle_filter(
        ps5.AppearanceModelPF,  # particle filter model class
        os.path.join(input_dir, "pres_debate"),
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect,
    )  # Add more if you need to


def part_4():
    template_rect = {"x": 210, "y": 37, "w": 103, "h": 285}

    save_frames = {
        40: os.path.join(output_dir, "ps5-4-a-1.png"),
        100: os.path.join(output_dir, "ps5-4-a-2.png"),
        240: os.path.join(output_dir, "ps5-4-a-3.png"),
        300: os.path.join(output_dir, "ps5-4-a-4.png"),
    }

    num_particles = 1200  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 12  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.0
    run_particle_filter(
        ps5.MDParticleFilter,
        os.path.join(input_dir, "pedestrians"),
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect,
        template_scale_low=0.982,
        template_scale_high=1.00,
        occ_thresh_1=4100,
        occ_thresh_2=3600,
        occ_time_1=100,
        occ_time_2=175,
        occ_dyn_scale=0.05,
    )  # Add more if you need to


def part_5():
    """Tracking multiple Targets.

    Use either a Kalman or particle filter to track multiple targets
    as they move through the given video.  Use the sequence of images
    in the TUD-Campus directory.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    template_rect1 = {"x": 80, "y": 150, "w": 80, "h": 150}
    template_rect2 = {"x": 295, "y": 220, "w": 40, "h": 90}
    template_rect3 = {"x": 0, "y": 200, "w": 50, "h": 110}

    save_frames = {
        28: os.path.join(output_dir, "ps5-5-a-1.png"),
        55: os.path.join(output_dir, "ps5-5-a-2.png"),
        70: os.path.join(output_dir, "ps5-5-a-3.png"),
    }
    # save_frames = {300: os.path.join(output_dir, 'test.png')}

    num_particles = 1200  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.00  # Set a value for alpha

    run_particle_filter_5(
        ps5.MDParticleFilter,  # particle filter model class
        os.path.join(input_dir, "TUD-Campus"),
        # input video
        template_rect1,
        template_rect2,
        template_rect3,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_scale_low=0.99,
        template_scale_high=1.01,
        occ_thresh_1=5000,
        occ_thresh_2=3600,
        occ_time_1=10,
        occ_time_2=175,
        occ_dyn_scale=0.10,
    )


def part_6():
    """Tracking pedestrians from a moving camera.

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    raise NotImplementedError


if __name__ == "__main__":
    # part_1b()
    # part_1c()
    # part_2a()
    # part_2b()
    # part_3()
    # part_4()
    part_5()
    # part_6()
