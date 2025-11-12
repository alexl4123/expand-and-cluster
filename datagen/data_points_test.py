import torch
import numpy as np

from datagen.regions_teacher_aware_data_generation import RegionsAwareDataGeneration 

class DataPointsTest:

    @classmethod
    def compute_parameters_student_with_overparameterization(cls, model, overparameterization_factor = 1.0):

        d_in = None

        parameters = 0
        for layer_index in range(len(model.fc_layers)):
            layer_ = model.fc_layers[layer_index].weight.data
            overparameterized_neurons = layer_.shape[0] * overparameterization_factor

            print(layer_index)
            print(layer_.shape)

            if layer_index == 0:
                prev_dimension = layer_.shape[1]
                d_in = prev_dimension
            else:
                prev_dimension = layer_.shape[1] * overparameterization_factor

            parameters += (prev_dimension + 1) * overparameterized_neurons 

        layer_ = model.fc.weight.data

        print(layer_.shape)
        print(len(layer_.shape))

        if len(layer_.shape) > 1:
            overparameterization_factor = 1 # As in 1p0
            output_neurons = layer_.shape[0]
            prev_dimension = layer_.shape[1] * overparameterization_factor
        else:
            overparameterization_factor = 1 # As in 1p0
            output_neurons = 1
            prev_dimension = layer_.shape[0] * overparameterization_factor

        parameters += (prev_dimension + 1) * output_neurons

        return d_in, parameters



    @classmethod
    def teacher_aware_uniform_datapoints_tests_o1p0_p0p5(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = parameters * 0.5
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(np.floor(parameters)),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)


    @classmethod
    def teacher_aware_uniform_datapoints_tests_o1p0_p1p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = parameters
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)



    @classmethod
    def teacher_aware_uniform_datapoints_tests_o1p0_p1p1(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = parameters * 1.1
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(np.ceil(parameters)),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)

    @classmethod
    def teacher_aware_uniform_datapoints_tests_o4p0_p0p5(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = parameters * 0.5
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(np.floor(parameters)),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)

    @classmethod
    def teacher_aware_uniform_datapoints_tests_o4p0_p1p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = parameters
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)


    @classmethod
    def teacher_aware_uniform_datapoints_tests_o4p0_p1p1(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = parameters * 1.1
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(np.ceil(parameters)),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)

    @classmethod
    def teacher_aware_uniform_datapoints_tests_o4p0_p5p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = parameters * 5.0
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        #data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(parameters),int(d_in)))
        data_points = np.random.uniform(-np.sqrt(3),np.sqrt(3),(int(np.ceil(parameters)),int(d_in)))

        return torch.tensor(data_points, dtype=torch.float32)

    @classmethod
    def teacher_aware_regions_datapoints_tests_o1p0_p0p5(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = int(np.floor(parameters * 0.5))
        if parameters < 2:
            parameters = 2
        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)




    @classmethod
    def teacher_aware_regions_datapoints_tests_o1p0_p1p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = parameters * 1.0
        if parameters < 2.0:
            parameters = 2.0
        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)


    @classmethod
    def teacher_aware_regions_datapoints_tests_o1p0_p1p1(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=1.0)

        parameters = int(np.ceil(parameters * 1.1))
        if parameters < 2:
            parameters = 2

        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)




    @classmethod
    def teacher_aware_regions_datapoints_tests_o4p0_p0p5(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = int(np.floor(parameters * 0.5))
        if parameters < 2:
            parameters = 2

        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)




    @classmethod
    def teacher_aware_regions_datapoints_tests_o4p0_p1p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = parameters
        if parameters < 2.0:
            parameters = 2.0

        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)


    @classmethod
    def teacher_aware_regions_datapoints_tests_o4p0_p1p1(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = int(np.ceil(parameters * 1.1))
        if parameters < 2:
            parameters = 2

        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)

    @classmethod
    def teacher_aware_regions_datapoints_tests_o4p0_p5p0(cls, augment = None, d_in = None, model = None):

        d_in, parameters = cls.compute_parameters_student_with_overparameterization(model, overparameterization_factor=4.0)

        parameters = int(np.ceil(parameters * 5.0))
        if parameters < 2:
            parameters = 2

        print(f"parameters: {parameters}")

        plan = cls.generate_model_plan_from_torch_model(model)
        dto = {}
        dto["teacher_specification"] = plan
        teacher_layers = cls.convert_teacher_layers_to_numpy_list(model)

        data_points, trace = RegionsAwareDataGeneration.generate_data_regions_heuristic(dto, teacher_layers, int(parameters))

        return torch.tensor(data_points, dtype=torch.float)





    @classmethod
    def generate_model_plan_from_torch_model(cls, model):
        # Assumed model is mnist_lenet, or fully_connected_specified

        plan = []
        d_in = None

        parameters = 0
        for layer_index in range(len(model.fc_layers)):
            layer_ = model.fc_layers[layer_index].weight.data

            if layer_index == 0:
                prev_dimension = layer_.shape[1]


                # We always assume to have a bias
                plan.append((True,layer_.shape[1]))
                plan.append((True,layer_.shape[0]))

            else:
                plan.append((True,layer_.shape[0]))

        # For our matters we only take 1 neuron as the last layer:
        #plan.append((True, 1))

        return plan


    @classmethod
    def convert_teacher_layers_to_numpy_list(cls, model):
        # Assumed model is mnist_lenet, or fully_connected_specified

        teacher_layers = []
        d_in = None

        parameters = 0
        for layer_index in range(len(model.fc_layers)):
            layer_ = ((model.fc_layers[layer_index].weight.data).detach().numpy())
            bias_  = (model.fc_layers[layer_index].bias.data).detach().numpy()

            new_layer = np.hstack((layer_, bias_[:,np.newaxis])) 

            teacher_layers.append(new_layer)


        layer_ = model.fc.weight.data
        bias_ = model.fc.bias.data
        new_layer = np.hstack((layer_, bias_[:,np.newaxis])) 

        if new_layer.shape[0] > 1:
            # Take any output neuron
            new_layer = (new_layer[0,:])

        teacher_layers.append(new_layer)

        return teacher_layers
