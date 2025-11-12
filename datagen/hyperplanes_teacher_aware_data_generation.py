
import numpy as np
import scipy
import torch

class HyperplanesTeacherAwareGeneration:

    @classmethod
    def hyperplanes_teacher_aware_data(cls, augment = None, d_in = None, model = None):
        """
        Only works for shallow teachers (1-hidden-layer)
        Assuming overparameterization of 4 (for #Data)
        """

        print("MADE IT TO HYPERPLANES AWARE GENERATION")
        print(model)

        first_layer = model.fc_layers[0]

        first_layer_weights = first_layer.weight.data
        first_layer_bias = first_layer.bias.data

        overparameterization_factor = 4
        overparameterized_neurons = first_layer_weights.shape[0] * overparameterization_factor
        input_dimension = first_layer_weights.shape[1]

        print(input_dimension)
        print(overparameterized_neurons)

        datapoints_to_generate = (input_dimension + 1) * overparameterized_neurons + (overparameterized_neurons * 10)

        print(first_layer_weights.shape)
        print(first_layer_bias.shape)
        print(first_layer_weights.dtype)
        first_layer = (first_layer_weights).clone().detach().numpy()
        first_layer_bias = (first_layer_bias).clone().detach().numpy()

        remaining_datapoints_to_generate = datapoints_to_generate

        points_per_hyperplane = int(remaining_datapoints_to_generate / first_layer_weights.shape[0])

        points = []

        while remaining_datapoints_to_generate > 0:
            # Ensures we generate as many datapoints as needed

            for neuron_index in range(first_layer.shape[0]):
                if remaining_datapoints_to_generate % 100 == 0:
                    print(remaining_datapoints_to_generate)


                # Per hyperplane we try to get 4 points
                no_bias_neuron = first_layer[neuron_index, :]
                bias = first_layer_bias[neuron_index]

                #denominator = (np.linalg.norm(neuron[0:(neuron.shape[0] - 1)]) ** 2)
                denominator = (np.dot(no_bias_neuron, no_bias_neuron))

                distances_to_hyperplane = np.random.normal(loc=0.0,scale=1.0,size=(points_per_hyperplane))

                numerators = (distances_to_hyperplane - bias)
                scale_facts = numerators / denominator
                A = no_bias_neuron.reshape(1, -1)
                orthogonal_basis = scipy.linalg.null_space(A).T

                if points_per_hyperplane > 1:
                    # Generate random shifts once
                    random_nullspace_shifts = np.random.normal(0.0, scale=1.0, size=(points_per_hyperplane, orthogonal_basis.shape[0]))

                    #print(orthogonal_basis.shape)
                    #print(random_nullspace_shifts.shape)

                    shift_vectors = np.dot(random_nullspace_shifts, orthogonal_basis)
                    #print(shift_vectors.shape)

                    scaled_points = no_bias_neuron * scale_facts[:, np.newaxis]
                    #print(scaled_points.shape)

                    points_local = scaled_points + shift_vectors

                    points.extend(points_local)

                    remaining_datapoints_to_generate -= points_local.shape[0]

                else:
                    for scale_fact_index in range(points_per_hyperplane):

                        # Find scaling factor s.t. point is exactly 'distance' away
                        scaled_point = no_bias_neuron * scale_facts[scale_fact_index]

                        # Generate random orthogonal vector (null-space vector)

                        random_nullspace_shift = np.random.normal(0.0, scale=1.0, size=(orthogonal_basis.shape[0]))

                        shift_vector = orthogonal_basis * random_nullspace_shift[:, np.newaxis]

                        shift_vector = np.sum(shift_vector, axis=0)
                        
                        # Shift vector is orthogonal to scaled point
                        point = scaled_point + shift_vector
                        points.append(point)

                        remaining_datapoints_to_generate -= 1

                        if remaining_datapoints_to_generate == 0:
                            break

                if remaining_datapoints_to_generate == 0:
                    break

            # IMPORTANT AFTER FIRST (FAST) ITERATION 
            points_per_hyperplane = 1

        points = np.array(points)

        print(points.shape)
        print(datapoints_to_generate)
        print("TEACHER AWARE DATA GENERATION FINISHED")
        print("-------------------")

        return torch.tensor(points, dtype=torch.float32)