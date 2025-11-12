import numpy as np


import torch.nn as nn
import torch


class SimpleNN(nn.Module):
    def __init__(self, plan, activation_function):
        super(SimpleNN, self).__init__()

        if activation_function == "ReLU" or activation_function == "relu":
            self.activation_function = nn.ReLU()
        elif activation_function == "g":
            self.activation_function = g()
        else:
            print(f"ACTIVATION FUNCTION {activation_function} NOT IMPLEMENTED!")
            exit(1)

        self.layers = []

        for layer_index in range(1,len(plan)):
            prev_size = plan[layer_index - 1][1]
            cur_size = plan[layer_index][1]
            self.layers.append(nn.Linear(prev_size, cur_size))

        last_size = plan[len(plan) - 1][1]
        self.output = nn.Linear(last_size, 1) # Assuming always 1 as output

    def forward(self, x):
        for layer in self.layers:
            # Single layer forward pass:
            x = layer(x)
            x = self.activation_function(x)
        x = self.output(x)
        return x



class RegionsAwareDataGeneration:

    gradient_threshold = 0.01

    step_normalized_size = 0.3
    max_steps = 15
    step_size_increase = 1.1

    init_area = 5.0
    glob_step_size = 1.0
    min_dist_break = 0.3
    np.random.seed(10000)


    @classmethod
    def g(cls):
        return _g

    @classmethod
    def _g(cls, x):
        return torch.sigmoid(4 * x) + F.softplus(x)

    @classmethod
    def points_permuted_adder(cls, position, points): # Add permutation to rmv. symmetries

        position = position.clone().detach().numpy()
        perturbation = np.random.uniform(low=-0.001,high=0.001, size=position.shape)
        position = position + perturbation
        points.append(position)


    @classmethod
    def fill_network_weights(cls, model, teacher_specification, teacher_layers):

        for layer_index in range(len(model.layers)):

            if teacher_specification[layer_index + 1][0] is True: # Bias:
                last_index = teacher_layers[layer_index].shape[1] - 1
                layer_no_bias = teacher_layers[layer_index][:,0:last_index]
                bias = teacher_layers[layer_index][:,last_index]
            else:
                last_index = teacher_layers[layer_index].shape[1] - 1
                layer_no_bias = teacher_layers[layer_index]
                bias = np.zeros(last_index)
            
            model.layers[layer_index].weight.data = torch.tensor(layer_no_bias, requires_grad=False, dtype=torch.float)
            model.layers[layer_index].bias.data = torch.tensor(bias, requires_grad=False, dtype=torch.float)

        # Assuming HERE with bias:
        last_layer = teacher_layers[len(teacher_layers) - 1]
        if len(last_layer.shape) == 1:
            last_layer_no_bias = last_layer[:-1]
            last_layer_bias = last_layer[-1]
        else:
            last_layer_no_bias = last_layer[0,:-1]
            last_layer_bias = last_layer[0,-1]

        model.output.weight.data = torch.tensor([last_layer_no_bias], requires_grad=False, dtype=torch.float)
        model.output.bias.data = torch.tensor([last_layer_bias], requires_grad=False, dtype=torch.float)

    @classmethod
    def measure_distance(cls, tensor_1, tensor_2):
        cur_dist = (tensor_1.clone().detach() - tensor_2.clone().detach()).pow(2).sum().sqrt()
        return cur_dist

    @classmethod
    def cosine_distance(cls, tensor_1, tensor_2):
        cosine_dist = spatial.distance.cosine(tensor_1.cpu().detach().numpy(), tensor_2.cpu().detach().numpy())
        return cosine_dist

    @classmethod
    def gradient_change_search(cls, model, pos_x, pos_grad, step, verbosity = 0, trace = []):

        step_size = cls.glob_step_size
        # POSITIVE SEARCH:
        boundary_found = False

        pos_x = pos_x.clone().detach().requires_grad_(True)
        output = model(pos_x)
        output.backward()
        pos_grad = pos_x.grad.clone().detach()
        pos_x = pos_x.clone().detach().requires_grad_(True)
        pos_x_prev = None

        for i in range(cls.max_steps):

            output = model(pos_x)
            output.backward()
            cur_grad = pos_x.grad

            cur_dist = cls.measure_distance(pos_grad, cur_grad)
            #cur_dist = cosine_distance(pos_grad, cur_grad)

            pos_x.grad.zero_()

            if verbosity == 1:
                print("-----")
                print(cur_grad)
                print(cur_dist)
                print("----")

            if cur_dist > cls.gradient_threshold:
                boundary_found = True
                break

            pos_x_prev = pos_x.clone().detach().requires_grad_(True)
            pos_x = (pos_x + step_size * step).clone().detach().requires_grad_(True)
            if trace is not None:
                trace.append(pos_x)
            step_size = step_size * cls.step_size_increase

        if pos_x_prev is None:
            print("[WARNING] -> POS-X-PREV IS NONE HANDLE ACTIVATED")
            pos_x_prev = (pos_x + step_size * step).clone().detach()
            boundary_found = False

        return pos_x, pos_x_prev, boundary_found

    @classmethod
    def binary_boundary_search(cls, model, end_pos, prev_pos):

        # POSITIVE SEARCH:
        boundary_found = False

        initial_pos = prev_pos.clone().detach().requires_grad_(True)
        output = model(initial_pos)
        output.backward()
        initial_grad = initial_pos.grad   


        end_pos = end_pos.clone().detach().requires_grad_(True)
        prev_pos = prev_pos.clone().detach().requires_grad_(True)

        for i in range(cls.max_steps):
            output = model(end_pos)
            output.backward()
            end_grad = end_pos.grad

            output = model(prev_pos)
            output.backward()
            prev_grad = prev_pos.grad

            middle_pos = ((end_pos + prev_pos) / 2).clone().detach().requires_grad_(True)
            output = model(middle_pos)
            output.backward()
            middle_grad = middle_pos.grad

            initial_middle_dist = cls.measure_distance(middle_grad, initial_grad)
            end_dist = cls.measure_distance(middle_grad, end_grad)
            prev_dist = cls.measure_distance(middle_grad, prev_grad)

            if end_dist < cls.gradient_threshold and prev_dist < cls.gradient_threshold:
                #print("===> (+++) FINAL GRAD DIST BOUNDARY SEARCH")
                break


            if initial_middle_dist < cls.gradient_threshold:
                prev_pos = middle_pos
            else:
                end_pos = middle_pos

            prev_end_dist = cls.measure_distance(prev_grad, end_grad)
            if prev_end_dist < cls.gradient_threshold:
                print("===> (!!!) FINAL GRAD DIST BOUNDARY SEARCH:")
                #print(prev_end_dist)
                break
            if cls.measure_distance(prev_pos, end_pos) < cls.min_dist_break:
                #print("===> FINAL GRAD DIST BOUNDARY SEARCH:")

                #print(prev_end_dist)
                break # Numerical reasons, that the points are not too close

        return prev_pos.clone().detach(), end_pos.clone().detach()

    @classmethod
    def compute_step(cls, model, inp, d_in, prev_step, keep_step_fixed = True):

        output = model(inp)
        output.backward()
        pos_grad = inp.grad.clone().detach()

        if keep_step_fixed is False:
            step = torch.tensor(np.random.uniform(low=-1.0,high=1.0, size=d_in), requires_grad=True, dtype=torch.float)
            #print("RANDOMLY GENERATED STEP")
            #print(step)
            step_norm = step.pow(2).sum().sqrt()
            step = (cls.step_normalized_size / step_norm) * step
            #print(step_norm)
            #print(cls.step_normalized_size)
            #print(step)
            #print("----")
        else:
            step = prev_step

        return pos_grad, step

    @classmethod
    def generation_helper(cls, model, pos_x, pos_grad, step, points, positive_active, datapoints_to_generate, d_in, trace = []):
        end_pos, prev_pos, boundary_found = cls.gradient_change_search(model, pos_x, pos_grad, step, verbosity=0, trace = trace)
        if boundary_found is True:
            # Add 2 points at the boundary
            pos_1, pos_2 = cls.binary_boundary_search(model, end_pos.clone().detach(), prev_pos.clone().detach())
            cls.points_permuted_adder(pos_1, points)
            #print("BOUNDARY POINT ADDED")
            #print(pos_1)
            #print("--- GRADS ---")
            #print(end_pos)
            #print(prev_pos)
            #print(pos_grad)

            if len(points) < datapoints_to_generate:
                cls.points_permuted_adder(pos_2, points)

            #end_pos = (end_pos + 3*step).clone().detach().requires_grad_(True)
            end_pos = (end_pos).clone().detach().requires_grad_(True)
            pos_grad, tmp_step = cls.compute_step(model, end_pos, d_in, step, keep_step_fixed = True)
            #print(pos_grad)
            #print("====")

            #cosine_dist = cosine_distance(step, tmp_step)
            #if cosine_dist > 0.300: # We do not want to go back ...
            #if cosine_dist > 0.8 and cosine_dist < 1.2: # We do not want to go back ...
            #    positive_active = False
            #elif cosine_dist >= 1.2:
            #    step = -tmp_step
            #else:
            #    step = tmp_step

        else:
            #print("OUTER POINT ADDED:")
            #print(end_pos)
            #print("===")
            cls.points_permuted_adder(end_pos, points)
            positive_active = False

        return end_pos, pos_grad, step, positive_active


    @classmethod
    def generate_data_regions_heuristic(cls, data_transfer_object, teacher_layers, datapoints_to_generate):

        teacher_specification = data_transfer_object["teacher_specification"]
        # Instantiate the model
        #activation_function = data_transfer_object["activation_function"]
        activation_function = "ReLU"

        model = SimpleNN(teacher_specification, activation_function)
        cls.fill_network_weights(model, teacher_specification, teacher_layers)

        model.eval() # Put model into evaluation mode (no update)

        points = []

        remaining_datapoints_to_generate = datapoints_to_generate

        # STARTING PROCEDURE:

        trace = None

        while len(points) < datapoints_to_generate:

            d_in = teacher_specification[0][1]
            inp = list(np.random.uniform(low=-cls.init_area,high=cls.init_area, size=d_in))
            inp = torch.tensor(inp, requires_grad=True, dtype=torch.float)

            positive_active = True
            negative_active = True

            #print("--- INIT POINT AT POSITION: ---")
            #print(inp)
            #print("")



            pos_x = inp.clone().detach().requires_grad_(True)
            neg_x = inp.clone().detach().requires_grad_(True)

            pos_grad, step = cls.compute_step(model, inp, d_in, None, keep_step_fixed=False)
            neg_step = (-step).clone().detach()
            neg_grad = (pos_grad).clone().detach()

            #print("-- STEP DIRECTION --")
            #print(step)
            #print("---")

            while len(points) < datapoints_to_generate and (positive_active is True or negative_active is True):

                if positive_active is True:
                    pos_x, pos_grad, step, positive_active = cls.generation_helper(model, pos_x, pos_grad, step, points, positive_active, datapoints_to_generate, d_in, trace = trace)

                if len(points) >= datapoints_to_generate:
                    break
                
                if negative_active is True:
                    neg_x, neg_grad, neg_step, negative_active = cls.generation_helper(model, neg_x.clone().detach().requires_grad_(True), neg_grad.clone().detach(), neg_step, points, negative_active, datapoints_to_generate, d_in, trace = trace)


        #while remaining_datapoints_to_generate > 0:
        #points = torch.tensor(points, dtype=torch.float)
        #print(points)
        return points, trace


