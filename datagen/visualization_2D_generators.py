import torch
import math
import os

import numpy as np

def before_after_decorator(method):
    def wrapper(cls, *args, **kwargs):

        if "model" in kwargs and "output_location" in kwargs:
            model = kwargs.get("model")
            output_location = kwargs.get("output_location")

            teacher_neurons = []
            for neuron_index in range(model.fc_layers[0].weight.shape[0]):
                weights = model.fc_layers[0].weight.detach().numpy()[neuron_index,:]
                biases = model.fc_layers[0].bias.detach().numpy()[neuron_index]

                params = list(weights)
                params.append(float(biases))

                teacher_neurons.append(params)

            print(teacher_neurons)
            np_array = np.array(teacher_neurons)

            np.savetxt(f"{output_location}teacher_neurons.csv", np_array, delimiter=',')
            
        result = method(cls, *args, **kwargs)

        teacher_query = result.detach().numpy()

        teacher_query_path = f"{output_location}teacher_query.csv"
        if os.path.isfile(teacher_query_path):
            student_data = np.genfromtxt(teacher_query_path, delimiter=',', skip_header=0)
            teacher_query = np.vstack([student_data, teacher_query])

        np.savetxt(teacher_query_path, teacher_query, delimiter=',')

        return result
    return wrapper

class Visualization2DGenerators:

    @classmethod
    @before_after_decorator
    def viz_uniform_m2_2_N500(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 500 
        input_data = torch.tensor(np.random.uniform(-2, 2, (num_samples, 2)), dtype=torch.float32)

        return input_data

    @classmethod
    @before_after_decorator
    def viz_uniform_m4_4_N500(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 500 
        input_data = torch.tensor(np.random.uniform(-4, 4, (num_samples, 2)), dtype=torch.float32)

        return input_data

    @classmethod
    @before_after_decorator
    def viz_uniform_m4_4_N1000(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 1000 
        input_data = torch.tensor(np.random.uniform(-4, 4, (num_samples, 2)), dtype=torch.float32)

        return input_data


    @classmethod
    @before_after_decorator
    def viz_uniform_m8_8_N500(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 500 
        input_data = torch.tensor(np.random.uniform(-8, 8, (num_samples, 2)), dtype=torch.float32)

        return input_data

    @classmethod
    @before_after_decorator
    def viz_gaussian_per_teacher_hyperplane(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 500 

        weights = model.fc_layers[0].weight.detach().numpy()
        biases = model.fc_layers[0].bias.detach().numpy()

        X = []

        for row_index in range(weights.shape[0]):
            row = weights[row_index, :]
            
            factor_zero_hyperplane = (-biases[row_index]) / np.dot(row, row)
            factor_1p5_hyperplane = (1.5) / np.dot(row, row)

            diff = row * factor_1p5_hyperplane
            normed_diff = np.linalg.norm(diff, ord=2) ** 2

            covariance = np.array([[normed_diff, 0.0], [0.0, normed_diff]])
            mean = np.array([0, 0])


            n_samples = 1000
            samples = np.random.multivariate_normal(mean, covariance, n_samples)

            samples = torch.tensor(samples, dtype=torch.float32)
            X.append(samples)


        print(X)
        X = torch.concat(X)

        return X

    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane(cls, augment = None, d_in = None, model=None,
                                              output_location = "", n_samples = 100,
                                              hyperplane_factor = 1.5, circle_gaussian_factor = 0.0):

        # Scales the cov-mat elements.
        # circle_gaussian_factor = 0, points are along a line.
        # circle_gaussian_factor = 1, gaussian according to all directions.
        scale_factor = 1 - circle_gaussian_factor

        weights = model.fc_layers[0].weight.detach().numpy()
        biases = model.fc_layers[0].bias.detach().numpy()

        X = []

        for row_index in range(weights.shape[0]):
            row = weights[row_index, :]
            
            factor_zero_hyperplane = (-biases[row_index]) / np.dot(row, row)
            factor_second_point = hyperplane_factor / np.dot(row, row)

            if np.linalg.norm(factor_second_point - factor_zero_hyperplane) < 1e-3:
                factor_second_point = float(np.random.uniform(low=-10, high=10, size=1)) / np.dot(row,row)

            w_zero = row * factor_zero_hyperplane
            w_second = row * factor_second_point

            data = np.array([
                w_zero,
                w_second
            ])

            cov_mat = np.cov(data.T)

            factor_1p5_hyperplane = hyperplane_factor / np.dot(row, row)

            diff = row * factor_1p5_hyperplane
            normed_diff = np.linalg.norm(diff, ord=2) ** 2

            if normed_diff < 1e-3:
                continue
            print(row)
            print(normed_diff)
            print(cov_mat)

            cov_factor = normed_diff / (np.max(cov_mat) + 1e-6)

            cov_mat = cov_factor * cov_mat
            #covariance = np.array([[normed_diff, 0.0], [0.0, normed_diff]])
            #mean = np.array([0, 0])

            identity_matrix = np.eye(cov_mat.shape[0])
            off_diag_mask = np.ones_like(cov_mat) - identity_matrix
            scaled_cov_mat = cov_mat * off_diag_mask * scale_factor + cov_mat * identity_matrix


            samples = np.random.multivariate_normal(w_zero, scaled_cov_mat, n_samples)

            samples = torch.tensor(samples, dtype=torch.float32)
            X.append(samples)


        #print(X)
        X = torch.concat(X)

        return X
    

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane(cls, augment = None, d_in = None, model=None,
                                              output_location = "", n_samples = 100,
                                              hyperplane_factor = 1.5, circle_gaussian_factor = 0.0,
                                              abs_offset = 0):

        # Scales the cov-mat elements.
        # circle_gaussian_factor = 0, points are along a line.
        # circle_gaussian_factor = 1, gaussian according to all directions.
        scale_factor = 1 - circle_gaussian_factor

        weights = model.fc_layers[0].weight.detach().numpy()
        biases = model.fc_layers[0].bias.detach().numpy()

        X = []

        for row_index in range(weights.shape[0]):
            row = weights[row_index, :]
            b = biases[row_index]

            x0 = row[0]
            x1 = row[1]


            angle = math.atan2(x1,x0)
            epsilon = 1e-6

            c = (abs_offset - b) / (x0 ** 2 + x1 ** 2 + epsilon)

            tail_x0 = x0 * c
            tail_x1 = x1 * c 
            w_tail = np.array([tail_x0, tail_x1])

            angle_p = angle + (np.pi / 2)
            angle_m = angle - (np.pi / 2)

            hyperplane_tail_x0 = tail_x0 + np.cos(angle_p)
            hyperplane_tail_x1 = tail_x1 + np.sin(angle_p)
            hyperplane_tip_x0 = tail_x0 + np.cos(angle_m)
            hyperplane_tip_x1 = tail_x1 + np.sin(angle_m) 

            w_zero = np.array([hyperplane_tip_x0, hyperplane_tip_x1])
            w_second = np.array([hyperplane_tail_x0, hyperplane_tail_x1])
            
            data = np.array([
                w_zero,
                w_second
            ])

            cov_mat = np.cov(data.T)

            factor_1p5_hyperplane = hyperplane_factor / np.dot(row, row)

            diff = row * factor_1p5_hyperplane
            normed_diff = np.linalg.norm(diff, ord=2) ** 2

            if normed_diff < 1e-3:
                continue

            print(row)
            print(normed_diff)
            print(cov_mat)

            cov_factor = normed_diff / (np.max(cov_mat) + 1e-6)
            cov_mat = cov_factor * cov_mat

            identity_matrix = np.eye(cov_mat.shape[0])
            off_diag_mask = np.ones_like(cov_mat) - identity_matrix
            scaled_cov_mat = cov_mat * off_diag_mask * scale_factor + cov_mat * identity_matrix

            samples = np.random.multivariate_normal(w_tail, scaled_cov_mat, n_samples)

            samples = torch.tensor(samples, dtype=torch.float32)
            X.append(samples)


        #print(X)
        X = torch.concat(X)

        return X




    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N100_HF1p5_CGF0(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location)


    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N100_HF1p5_CGF0p05(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.05,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location)


    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N100_HF15_CGF0(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0,
                                                         hyperplane_factor=15,
                                                         output_location=output_location)


    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N100_HF15_CGF0p05(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.05,
                                                         hyperplane_factor=15,
                                                         output_location=output_location)

    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N100_HF1p5_CGF1(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=1.0,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location)

    @classmethod
    @before_after_decorator
    def viz_gaussian_direction_teacher_hyperplane_N500_HF1p5_CGF1(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_direction_teacher_hyperplane(model=model,
                                                         n_samples=500,
                                                         circle_gaussian_factor=1.0,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0_ABSOF2(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.0,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=2)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0_ABSOFm2(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.0,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-2)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF2(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=2)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm2(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-2)


    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF20(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=20)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm20(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-20)


    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF50(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=50)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm50(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-50)




    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF100(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=100)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm100(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-100)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF1000(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=1000)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm1000(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-1000)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOF10000(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=10000)

    @classmethod
    @before_after_decorator
    def viz_gaussian_parallel_teacher_hyperplane_N100_HF0_CGF0p5_ABSOFm10000(cls, augment = None, d_in = None, model=None,
                                              output_location = ""):
        return cls.gaussian_parallel_teacher_hyperplane(model=model,
                                                         n_samples=100,
                                                         circle_gaussian_factor=0.5,
                                                         hyperplane_factor=1.5,
                                                         output_location=output_location,
                                                         abs_offset=-10000)


    @classmethod
    @before_after_decorator
    def viz_uniform_m20_20_N1000(cls, augment = None, d_in = None, model=None, output_location = ""):
        num_samples = 1000 
        input_data = torch.tensor(np.random.uniform(-20, 20, (num_samples, 2)), dtype=torch.float32)

        return input_data









