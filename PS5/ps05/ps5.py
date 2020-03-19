"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2
import time

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # i. State vector (X) with the initial x and y values.
        self.state = np.array([[init_x, init_y, 0.0, 0.0]]).T
        # ii. Covariance 4x4 array (Σ ) initialized with a diagonal
        # matrix with some value.
        P_uncert_pos = 0.1
        P_uncert_vel = 0.1
        self.P = np.diag([P_uncert_pos, P_uncert_pos, P_uncert_vel, P_uncert_vel])

        # iii. 4x4 state transition matrix D_t
        # change in time
        dt = 1.0
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # iv. 2x4 measurement matrix M_t
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # v. 4x4 process noise matrix Σd_t
        self.Q = Q
        # vi. 2x2 measurement noise matrix Σm_t
        self.R = R
        # 4x4 identity matrix
        self.I = np.identity(4)

    def predict(self):
        # @ operator is for matrix dot multiplication
        # https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
        # state prediction
        self.state = self.F @ self.state
        # covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, meas_x, meas_y):
        # @ operator is for matrix dot multiplication
        # https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
        # map measurements
        Z = np.array([[meas_x, meas_y]]).T
        # compute error
        Y = Z - self.H @ self.state
        # compute system uncertainty
        S = (self.H @ self.P @ self.H.T) + self.R
        # compute Kalmann gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        # state update
        self.state += K @ Y

        # uncertainty covariance update
        self.P = (self.I - (K @ self.H)) @ self.P

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)
        return self.state[0, 0], self.state[1, 0]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get("num_particles")  # required by the autograder
        self.sigma_exp = kwargs.get("sigma_exp")  # required by the autograder
        self.sigma_dyn = kwargs.get("sigma_dyn")  # required by the autograder
        self.template_rect = kwargs.get("template_coords")  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.template = self.get_gray_scale(template)
        self.frame = self.get_gray_scale(frame)
        self.frame_h, self.frame_w = frame.shape[:2]
        self.particles = self.init_particles(
            self.frame_h, self.frame_w, self.num_particles
        )
        self.weights = self.init_p_weights(self.num_particles)

    def get_gray_scale(self, frame):
        frame_R = frame[:, :, 0]
        frame_G = frame[:, :, 1]
        frame_B = frame[:, :, 2]
        frame = frame_R * 0.3 + frame_G * 0.58 + frame_B * 0.12
        return frame

    def init_particles(self, height, width, num_particles):
        """
        initializes particles with random values of height and width
        Args: 
            height (int): height of the image particle
            width (int): width of the image particle
        Returns (np.array(numpy.array)): a numpy array containing length 2 
                                         numpy array containing 
                                         random height and width values
        """
        particles = list()
        for i in range(num_particles):
            rnd_height = np.random.choice(height)
            rnd_width = np.random.choice(width)
            # append a length 2 numpy array containing
            # a random height and a random width
            particles.append(np.array([rnd_width, rnd_height]))
        return np.array(particles)

    def init_p_weights(self, num_particles):
        """
        initializes the particle weights to be uniform
        Args: 
            num_particles(int): number of particles
        Returns (np.array): numpy array containing weights
        """
        return np.ones(self.num_particles) / self.num_particles

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.
        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def calculate_mse(self, template, frame_cutout):
        # resize if necessary
        temp_h, temp_w = template.shape[:2]
        if frame_cutout.shape != template.shape:
            f_h, f_w = frame_cutout.shape[:2]
            f_y = float(temp_h) / float(f_h)
            f_x = float(temp_w) / float(f_w)
            frame_cutout = cv2.resize(frame_cutout, None, fx=f_x, fy=f_y)
        mse = np.sum((template.astype(np.float) - frame_cutout.astype(np.float)) ** 2)
        mse /= float(temp_h * temp_h)
        return mse

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.
           which is calculated using exp(-MSE/(2*sigma_mse**2))
        Returns:
            float: similarity value.
        """
        mse = self.calculate_mse(template, frame_cutout)

        return np.exp(-mse / (2.0 * self.sigma_exp ** 2.0))

    def get_patch_coord(self, particle):
        """
        Args: 
            particle (np.array): particle to be checked
        Returns(dict): a dictionary of patch if particle is within bounds
                       None otherwise
        """
        x, y = particle
        temp_h, temp_w = self.template.shape[:2]
        # y coordinates are reversed because of how rows are represented
        y_min = int(y - temp_h / 2)
        y_max = int(y + temp_h / 2)
        x_min = int(x - temp_w / 2)
        x_max = int(x + temp_w / 2)
        # limit particles to borders
        if y_min < 0:
            offset = 0 - y_min
            y_min = 0
            y_max += offset
        elif y_max > self.frame_h:
            offset = self.frame_h - y_max
            y_max = self.frame_h
            y_min += offset
        if x_min < 0:
            offset = 0 - x_min
            x_min = 0
            x_max += offset
        elif x_max > self.frame_w:
            offset = self.frame_w - x_max
            x_max = self.frame_w
            x_min += offset
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_particle_from_patch_coord(self, patch_coord):
        """
        Args:
            patch (dict): patch coordinates
        Returns (np.array): particle from patch coordinates
        """
        temp_h, temp_w = self.template.shape[:2]
        return np.array(
            [
                int(patch_coord["x_min"] + temp_w / 2),
                int(patch_coord["y_min"] + temp_h / 2),
            ]
        )

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        rsmp_particles = list()
        # resample with replacement from particles using np.random.choice
        # resamp_indicies, a np.array containing random indicies in
        # num particlces

        rsmp_indicies = np.random.choice(
            self.num_particles, size=self.num_particles, replace=True, p=self.weights
        )

        # continue resampling until number of particles is met
        for r_idx in range(self.num_particles):
            p_idx = rsmp_indicies[r_idx]
            rsmp = self.particles[p_idx]
            # boundary checking
            patch_coord = self.get_patch_coord(rsmp)
            new_particle = self.get_particle_from_patch_coord(patch_coord)
            r_x, r_y = rsmp
            p_x, p_y = new_particle
            rsmp_particles.append(new_particle)

        return np.array(rsmp_particles)

    def get_patch(self, particle):
        """
        returns the image patch of the particle
        Args: 
            particle (np.array): particle
        Returns (np.array): the image patch cooresponding to particle
        """
        patch_xy = self.get_patch_coord(particle)
        if patch_xy:
            y_max = patch_xy["y_max"]
            y_min = patch_xy["y_min"]
            x_max = patch_xy["x_max"]
            x_min = patch_xy["x_min"]
            return self.frame[y_min:y_max, x_min:x_max]
        else:
            return None

    def recalculate_weights(self):
        """
        recalculates the weights of the particlces
        """
        for i, particle in enumerate(self.particles):
            patch = self.get_patch(particle)
            err = self.get_error_metric(self.template, patch)
            self.weights[i] = err
        self.weights /= np.sum(self.weights)

    def update_dynamics(self):
        """
        updates particles using gaussian dynamics model
        """
        self.particles += np.random.normal(
            0, self.sigma_dyn, self.particles.shape
        ).astype(np.int)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        # 2 days worth of debugging because I forgot the following line
        # to update frames.
        self.frame = self.get_gray_scale(frame)
        # resample particles
        self.particles = self.resample_particles()

        # update dynamics using random gaussian
        self.update_dynamics()

        # calculate weights
        self.recalculate_weights()

    def get_eucld_dist(self, pt_1, pt_2):
        """
        Args:
            pt_1 (tuple): tuple of y and x of point 1
            pt_2 (tuple): tuple of y and x of point 2
        Returns (float): euclidian distance of point 1 and 2
        """
        pt_1_x, pt_1_y = pt_1
        pt_2_x, pt_2_y = pt_2
        return np.sqrt((pt_1_y - pt_2_y) ** 2 + (pt_1_x - pt_2_x) ** 2)

    def get_best_estimate_coord(self):
        """
        Calculates the best estimate coordinates
        Returns (tuple): x and y coordinates of the most weighted estimate
        """
        max_weight_idx = np.argmax(self.weights)
        x, y = self.particles[max_weight_idx]
        return (x, y)

    def get_mean_estimate_coord(self):
        """
        Calculates the best estimate coordinates
        Returns (tuple): x and y coordinates of the weighted mean estimate
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        return (x_weighted_mean, y_weighted_mean)

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean, y_weighted_mean = self.get_mean_estimate_coord()

        #  Every particle's (x, y) location in the distribution should be
        #  plotted by drawing a colored dot point on the image. Remember that
        #  this should be the center of the window, not the corner.
        RADIUS = 1
        THICKNESS = 2
        for i, (p_x, p_y) in enumerate(self.particles):
            c = int(self.weights[i] * 10000)
            COLOR = (c, c, c)
            cv2.circle(frame_in, (p_x, p_y), RADIUS, COLOR, THICKNESS)
        #  Draw the rectangle of the tracking window associated with the
        #  Bayesian estimate for the current location which is simply the
        #  weighted mean of the (x, y) of the particles.
        weighted_mean_xy = (int(x_weighted_mean), int(y_weighted_mean))
        COLOR = (255, 255, 255)
        rect_xy = self.get_patch_coord(weighted_mean_xy)
        # upper left point
        pt_1 = (rect_xy["x_min"], rect_xy["y_min"])
        # lower right point
        pt_2 = (rect_xy["x_max"], rect_xy["y_max"])
        cv2.rectangle(frame_in, pt_1, pt_2, COLOR, THICKNESS)
        #  Finally we need to get some sense of the standard deviation or
        #  spread of the distribution. First, find the distance of every
        #  particle to the weighted mean. Next, take the weighted sum of these
        #  distances and plot a circle centered at the weighted mean with this
        #  radius.
        COLOR = (200, 200, 0)
        weighted_sum_dist = 0
        for i, particle in enumerate(self.particles):
            dist = self.get_eucld_dist(weighted_mean_xy, particle)
            weighted_sum_dist += dist * self.weights[i]
        cv2.circle(
            frame_in,
            (int(x_weighted_mean), int(y_weighted_mean)),
            int(weighted_sum_dist),
            COLOR,
            THICKNESS,
        )


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """
        super(AppearanceModelPF, self).__init__(
            frame, template, **kwargs
        )  # call base class constructor

        self.alpha = kwargs.get("alpha")  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        # timer to update template
        self.time = 0

    def choose_estimate(self):
        """
        chooses the best esimation method based on time
        """
        best_est_xy = None
        if self.time > 3:
            best_est_xy = self.get_best_estimate_coord()
        elif self.time > 10:
            best_est_xy = self.get_mean_estimate_coord()

        self.time += 1
        return best_est_xy

    def update_template(self):
        """
        updates the template using
        Template(t) = alpha*Best(t) + (1 - alpha)Template(t-1)
        """
        best_est_xy = self.choose_estimate()
        if best_est_xy:
            best_est = self.get_patch(best_est_xy)

            # resize image if templates do not match
            if best_est.shape != self.template.shape:
                b_h, b_w = best_est.shape[:2]
                temp_h, temp_w = self.template.shape[:2]
                f_y = float(temp_h) / float(b_h)
                f_x = float(temp_w) / float(b_w)
                best_est = cv2.resize(best_est, None, fx=f_x, fy=f_y)
            self.template = self.alpha * best_est + (1.0 - self.alpha) * self.template

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        # 2 days worth of debugging because I forgot the following line
        # to update frames.
        self.frame = self.get_gray_scale(frame)
        # resample particles
        self.particles = self.resample_particles()

        # update dynamics using random gaussian
        self.update_dynamics()

        # calculate weights
        self.recalculate_weights()

        # update template
        self.update_template()


class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """
        super(MDParticleFilter, self).__init__(
            frame, template, **kwargs
        )  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.template_scale_low = kwargs.get("template_scale_low")
        self.template_scale_high = kwargs.get("template_scale_high")
        self.occ_thresh_1 = kwargs.get("occ_thresh_1")
        self.occ_time_1 = kwargs.get("occ_time_1")
        self.occ_thresh_2 = kwargs.get("occ_thresh_2")
        self.occ_time_2 = kwargs.get("occ_time_2")
        self.occ_dyn_scale = kwargs.get("occ_dyn_scale")
        self.templates = self.init_templates(self.template)

    def init_templates(self, template):
        """
        returns (np.array): initialized templates of different sizes
        """
        templates = list()
        # low and high for how much the image scaling is allowed to change

        for i in range(self.num_particles):
            scale = np.random.uniform(
                low=self.template_scale_low, high=self.template_scale_high
            )
            resized_template = cv2.resize(np.copy(template), None, fx=scale, fy=scale)
            templates.append(resized_template)
        return np.array(templates)

    def get_best_estimate(self):
        """
        Calculates the best estimate coordinates
        Returns (tuple): x and y coordinates of the most weighted estimate
        """
        max_weight_idx = np.argmax(self.weights)
        x, y = self.particles[max_weight_idx]
        return (x, y, max_weight_idx)

    def get_mean_estimate(self):
        """
        Calculates the best estimate coordinates
        Returns (tuple): x and y coordinates of the mean estimate
        """
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]
        # find the particle closest to the weighted mean
        dists = list()
        for i, particle in enumerate(self.particles):
            dists.append(
                self.get_eucld_dist((x_weighted_mean, y_weighted_mean), particle)
            )
        dists = np.array(dists)
        idx = np.argmin(dists)

        return (x_weighted_mean, y_weighted_mean, idx)

    def choose_estimate(self):
        """
        chooses the best esimation method based on time
        """
        best_est = None
        if self.time > 3:
            best_est = self.get_best_estimate()
        if self.time > 10:
            best_est = self.get_mean_estimate()
        self.time += 1
        return best_est

    def update_template(self):
        """
        updates the template using
        Template(t) = alpha*Best(t) + (1 - alpha)Template(t-1)
        """
        best_est = self.choose_estimate()

        if best_est:
            best_est_x, best_est_y, best_est_idx = best_est
            best_est = self.get_patch((best_est_x, best_est_y))
            self.template = self.templates[best_est_idx]
            # resize image if templates do not match
            if best_est.shape != self.template.shape:
                b_h, b_w = best_est.shape[:2]
                temp_h, temp_w = self.template.shape[:2]
                f_y = float(temp_h) / float(b_h)
                f_x = float(temp_w) / float(b_w)
                best_est = cv2.resize(best_est, None, fx=f_x, fy=f_y)
            self.template = self.alpha * best_est + (1.0 - self.alpha) * self.template

    def update_templates(self):
        """
        logic for updating templates in spite of occlusion
        """
        if self.time > 3:
            self.templates = self.init_templates(self.template)

    def recalculate_weights(self):
        """
        recalculates the weights of the particlces
        """
        mses = list()
        for i, particle in enumerate(self.particles):
            patch = self.get_patch(particle)
            err = self.get_error_metric(self.templates[i], patch)
            self.weights[i] = err
            mses.append(self.calculate_mse(self.templates[i], patch))
        self.weights /= np.sum(self.weights)

    def is_occlusion(self):
        """
        returns (bool): True if occlusion is detected by comparing
        err against a threshold
        """
        """
        recalculates the weights of the particlces
        """

        for i, particle in enumerate(self.particles):
            patch = self.get_patch(particle)
            mse = self.calculate_mse(self.template, patch)
            if mse > self.occ_thresh_1 and self.time > self.occ_time_1:
                return True
            elif mse > self.occ_thresh_2 and self.time > self.occ_time_2:
                return True
        return False

    def update_dynamics_occlusion(self):
        """
        updates particles using gaussian dynamics model
        """

        sigma_dyn = self.occ_dyn_scale * self.sigma_dyn
        self.particles += np.random.normal(0, sigma_dyn, self.particles.shape).astype(
            np.int
        )

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """

        # 2 days worth of debugging because I forgot the following line
        # to update frames.
        self.frame = self.get_gray_scale(frame)
        # if self.time >= 120:
        # cv2.imshow("img", cv2.convertScaleAbs(self.template))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # # only resample if not occlusion
        if not self.is_occlusion():
            # resample particles
            self.particles = self.resample_particles()

            # update dynamics using random gaussian
            self.update_dynamics()
            # update templates
            self.update_templates()

            # calculate weights
            self.recalculate_weights()

            # update template
            self.update_template()

        else:
            self.update_dynamics_occlusion()
            # update template
            self.update_template()
