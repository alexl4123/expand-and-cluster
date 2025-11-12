import yaml
import os

# Specify the folder name here (and name prefix):
name_prefix = "20240604"
start_index = 0

datagen_name = "teacher_aware_uniform_datapoints_tests_"
#datagen_name = "teacher_aware_regions_datapoints_tests_"

activation_functions = ["g"]
number_hidden_layers= [2]
overparameterization_factors = [4.0]

d_ins = [2,4,8]
biases = [True]
hidden_layer_sizes = [3]


template = {}

def format_float(value):
    # Format the float to one decimal place
    formatted_value = f"{value:.1f}"
    # Replace the dot with 'p'
    formatted_value = formatted_value.replace('.', 'p')
    return formatted_value


data_plan = [
        "5p0"
        ]

data_plan_string = "specification"

experiment = {
    "iterations_per_dataset": 7,
    # The following parameters don't really matter ...
    "parallelism": "none",
    "number_of_teachers": 1,
    "number_datasets": 1,
    "processors": 1,
    }
train_params = {
    "maxtime_ode": 3600,
    "maxtime_optim": 3600,
    "maxiterations_optim": 1e7,
    "maxiterations_ode": 1e10,
    "verbosity": 1
    }
data_gen = {
    "teacher_specification": "sample",
    "data_specification": "sample",
    "data_plan": "file",
    }

template["experiment"] = experiment
template["train_params"] = train_params
template["data_gen"] = data_gen

with open("template.txt", "r") as file:
    template_content = file.read()


for activation_function_index in range(len(activation_functions)):
    for number_hidden_layer_index in range(len(number_hidden_layers)):
        for overparameterization_index in range(len(overparameterization_factors)):

            counter = 0

            for d_in in d_ins:
                for bias in biases:
                    for hidden_layer_size in hidden_layer_sizes:

                        activation_function = activation_functions[activation_function_index]
                        number_hidden_layer = number_hidden_layers[number_hidden_layer_index]
                        overparameterization_factor = overparameterization_factors[overparameterization_index]
                        
                        teacher_architecture = []
                        student_architecture = []

                        for hidden_layer in range(number_hidden_layer):

                            teacher_architecture.append((hidden_layer_size,bias))
                            student_architecture.append((int(hidden_layer_size * overparameterization_factor), bias))

                        teacher_architecture = tuple(teacher_architecture)
                        student_architecture = tuple(student_architecture)

                        architecture = {
                            "d_in": d_in,
                            "teacher": {},
                            "student": {}
                            }
                        architecture["teacher"]["activation_function"] = activation_function
                        architecture["teacher"]["hidden_layers"] = number_hidden_layer

                        architecture["student"]["activation_function"] = activation_function
                        architecture["student"]["hidden_layers"] = number_hidden_layer

                        student_architecture_string = ""
                        teacher_architecture_string = ""

                        for layer_index in range(len(student_architecture)):
                            architecture["student"][f"number_neurons_{layer_index + 1}"] = student_architecture[layer_index][0]

                            student_architecture_string += f"_{student_architecture[layer_index][0]}"
                            teacher_architecture_string += f"_{teacher_architecture[layer_index][0]}"

                            architecture["student"][f"bias_{layer_index + 1}"] = student_architecture[layer_index][1]

                            architecture["teacher"][f"number_neurons_{layer_index + 1}"] = teacher_architecture[layer_index][0]
                            architecture["teacher"][f"bias_{layer_index + 1}"] = teacher_architecture[layer_index][1]

                        student_architecture_string  =f"students_fully_connected_specified({experiment['iterations_per_dataset']})" + student_architecture_string
                        teacher_architecture_string = f"fully_connected_specified{teacher_architecture_string}"
    

                        to_write_content = template_content


                        to_write_content = to_write_content.replace('<STUDENT-ARCHITECTURE>', student_architecture_string)
                        to_write_content = to_write_content.replace('<TEACHER-ARCHITECTURE>', teacher_architecture_string)

                        str_overparam = format_float(overparameterization_factors[overparameterization_index])
                        to_write_content = to_write_content.replace('<D_IN>', str(d_in))

                        template["architecture"] = architecture

                        name_postfix = f"{activation_function_index}{number_hidden_layer_index}{overparameterization_index}{counter}"

                        template["experiment"]["name"] = name_postfix
                        template["experiment"]["name_prefix"] = name_prefix


                        to_write_content = to_write_content.replace('<NAME>', f"{template['experiment']['name_prefix']}{template['experiment']['name']}")

                        full_name = f"{name_prefix}{name_postfix}"
                        os.makedirs(f"{name_prefix}", exist_ok=True)

                        for data_plan_index in range(len(data_plan)):
                            final_to_write_content = to_write_content
                            final_to_write_content = final_to_write_content.replace('<DATAGEN>', f"{datagen_name}o{str_overparam}_p{data_plan[data_plan_index]}")
                            final_to_write_content = final_to_write_content.replace('<EXP-DATA-INDEX>', f"{data_plan_index + 1}")

                            print(final_to_write_content)

                            with open(os.path.join(*[name_prefix,f"{full_name}_EC_STARTER_{data_plan_index + 1}.txt"]), 'w') as file:
                                file.write(final_to_write_content)


                        path_components = [name_prefix, f"{full_name}.yaml"]
                        file_path = os.path.join(*path_components)

                        with open(file_path, "w") as file:
                            yaml.dump(template, file, default_flow_style=False)

                        data_plan_string = "specification\n"
                        for data_plan_item in data_plan:
                            prefix = "student_parameters"

                            if data_plan_item == "m1":
                                data_plan_string += f"\"{prefix}-1\"\n"
                            elif data_plan_item == "p1":
                                data_plan_string += f"\"{prefix}+1\"\n"
                            elif data_plan_item == "1p0":
                                data_plan_string += f"\"{prefix}\"\n"
                            else:
                                data_plan_string += f"\"{prefix}*{data_plan_item}\"\n"

                        with open(os.path.join(*[name_prefix, f"{full_name}_data_plan.csv"]), "w") as file:
                            file.write(data_plan_string)

                        counter += 1

