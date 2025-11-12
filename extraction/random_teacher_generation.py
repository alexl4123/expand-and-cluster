
import numpy as np

import logging


from models import bn_initializers, initializers, activation_functions, fully_connected_specified, students_fully_connected_specified


logger = logging.getLogger('')

class RandomTeacherGeneration:

    @classmethod
    def random_teacher_generation(cls, d_in, teacher_specification, activation_function, precision = ""):

        # Select the activation function.
        if hasattr(activation_functions, activation_function):
            activation_function = getattr(activation_functions, activation_function)()
        else:
            raise ValueError('No activation function: {} (add it in models.activation_functions)'.format(
                model_hparams.act_fun))



        print("RANDOM MODEL GENERATION")

        print(teacher_specification)

        splits = teacher_specification.split("_")
        print(splits)

        if splits[0] == "fully" and splits[1] == "connected" and splits[2] == "specified":
            
            layer_description = splits[3:]

            print("MATCH")
            print(d_in)
            print(layer_description)

            plan = [int(i) for i in layer_description]

            weights, biases = cls.weights_initialization(plan, d_in)

            model = fully_connected_specified.Model(plan, None, activation_function,
                    weights, biases,
                    outputs=1, input_dimensions = d_in,
                    precision=precision)



        elif splits[0] == "students" and splits[1] == "fully" and splits[2] and "connected" and "specified" in splits[3]:

            N = int(splits[3][splits[3].find("(")+1:splits[3].find(")")])


            if N == "":
                N = 1

            layer_description = splits[4:]
            plan = [int(i) for i in layer_description]

            print("MATCH")
            print(N)
            print(d_in)
            print(layer_description)
            print(plan)

            weights, biases = cls.weights_initialization(plan, d_in, N = N)


            model_plan = [N] + plan

            model = students_fully_connected_specified.Model(
                    model_plan, None, activation_function,
                    students_weights=weights, students_biases=biases,
                    outputs=1, input_dimensions = d_in,
                    precision=precision)

        else:
            print("[ERROR] - MODEL NOT FOUND!")
            exit(0)

        

        return model

    @classmethod
    def weights_initialization(cls, weights_plan, d_in, N = None):

        if N is None:
            # Teacher Initialization:
            w_teach = []
            b_teach = []

            for layer in range(0, len(weights_plan)):
                if layer == 0:
                    neurons_layer = weights_plan[layer]
                    tmp_d_in = d_in
                    tmp_w_teach, tmp_b_teach = cls.weights_initialization_helper(tmp_d_in, neurons_layer, layer)
                else:
                    prev_layer = weights_plan[layer-1]
                    neurons_layer = weights_plan[layer]
                    tmp_w_teach, tmp_b_teach = cls.weights_initialization_helper(prev_layer, neurons_layer, layer)

                print("TEACHER LAYER")
                print(tmp_w_teach)
                w_teach.append(tmp_w_teach)
                b_teach.append(tmp_b_teach)


            # Generate a random 2D vector
            vec = np.random.randn(weights_plan[len(weights_plan) - 1])
            # Normalize the vector to have an L2 norm of 1
            vec /= np.linalg.norm(vec)
            # Scale the vector to have an L2 norm in the range [1.5, 2.5]
            norm_scale = np.random.uniform(1.5, 2.5)
            vec *= norm_scale
            random_bias = np.array([np.random.uniform(-np.linalg.norm(vec), np.linalg.norm(vec))])

            w_teach.append(np.array([vec.T]))
            b_teach.append(random_bias)

        else:
            # Student Initialization:
            w_teach = []
            b_teach = []


            for student_index in range(N):

                student_weight = []
                student_bias = []

                for layer in range(0, len(weights_plan)):
                    if layer == 0:
                        neurons_layer = weights_plan[layer]
                        tmp_d_in = d_in
                        tmp_w_teach, tmp_b_teach = cls.weights_initialization_helper(tmp_d_in, neurons_layer, layer)
                    else:
                        prev_layer = weights_plan[layer-1]
                        neurons_layer = weights_plan[layer]
                        tmp_w_teach, tmp_b_teach = cls.weights_initialization_helper(prev_layer, neurons_layer, layer)

                    print("TEACHER LAYER")
                    print(tmp_w_teach)
                    student_weight.append(tmp_w_teach)
                    student_bias.append(tmp_b_teach)


                # Generate a random 2D vector
                vec = np.random.randn(weights_plan[len(weights_plan) - 1])
                # Normalize the vector to have an L2 norm of 1
                vec /= np.linalg.norm(vec)
                # Scale the vector to have an L2 norm in the range [1.5, 2.5]
                norm_scale = np.random.uniform(1.5, 2.5)
                vec *= norm_scale
                random_bias = np.array([np.random.uniform(-np.linalg.norm(vec), np.linalg.norm(vec))])

                student_weight.append(np.array([vec.T]))
                student_bias.append(random_bias)

                w_teach.append(student_weight)
                b_teach.append(student_bias)



        return w_teach, b_teach


    @classmethod
    def weights_initialization_helper(cls, tmp_d_in, neurons_layer, layer):
        tmp_w_teach = None
        tmp_b_teach = None

        for _ in range(neurons_layer):
            # Generate a random 2D vector
            vec = np.random.randn(tmp_d_in)
            # Normalize the vector to have an L2 norm of 1
            vec /= np.linalg.norm(vec)
            # Scale the vector to have an L2 norm in the range [1.5, 2.5]
            norm_scale = np.random.uniform(1.5, 2.5)
            vec *= norm_scale
            # Account for bias:
            # Generate a random float in the range [-norm-of-vector, norm-of-vector]
            random_bias = np.random.uniform(-np.linalg.norm(vec), np.linalg.norm(vec))
            # Concatenate the bias as the third component to the vector
            #vec = np.append(vec, random_bias)

            if tmp_w_teach is None:
                tmp_w_teach = vec
                tmp_b_teach = random_bias
            else:
                tmp_w_teach = np.column_stack((tmp_w_teach, vec))
                tmp_b_teach = np.column_stack((tmp_b_teach, random_bias))
            

        tmp_w_teach = tmp_w_teach.T
        tmp_b_teach = tmp_b_teach[0,:]
        return tmp_w_teach, tmp_b_teach.T

