# Small script for creating configuration files.
# Replace (starting)time, date, number_students, etc. programmatically.
# -------------------------------------------------------------------------
# Other Info:
# SAME-SIZE-REQUIRED: teacher_array, samples_array, training_steps_array
# --> Works as an "outer loop", meaning that for each tuple (teacher_array[i], samples_array[i], training_steps_array[i]) one inner loop is called
# OTHER-SIZE-THAN-ABOVE: dataset_array
# --> Works as an "inner loop", therefore these datasets are called for one outer loop iteration
# -------------------------------------------------------------------------

hour="10"
minute="00"

date=20240813
number_students=10
teacher_seed=1
global_seed=1

template_file="configuration_template.txt"
#student_hidden_layer_size=2048

declare -a student_hidden_layer_array=()
#student_hidden_layer_array+=(16)

student_hidden_layer_array+=("128_128_128")
student_hidden_layer_array+=("128_128_128")
student_hidden_layer_array+=("128_128_128")
student_hidden_layer_array+=("128_128_128")

#student_hidden_layer_array+=(512)
#student_hidden_layer_array+=(1024)
#student_hidden_layer_array+=(2048)

declare -a teacher_array=()
#teacher_array+=("09438e1695")
#teacher_array+=("6aca1e87a7")
#teacher_array+=("531cde1a25")
#teacher_array+=("70fbf466d8")
#teacher_array+=("b4b9a61d8a")
#teacher_array+=("6d174a27e7")
#teacher_array+=("10c5cd6563")
#teacher_array+=("473714f244")

#teacher_array+=("593ba334b9")

teacher_array+=("e9b66ad22b")
teacher_array+=("e9b66ad22b")
teacher_array+=("e9b66ad22b")
teacher_array+=("e9b66ad22b")

#teacher_name="09438e1695"
#teacher_name="6aca1e87a7"
#teacher_name="531cde1a25"
#teacher_name="70fbf466d8"
#teacher_name="b4b9a61d8a"
#teacher_name="6d174a27e7"
#teacher_name="10c5cd6563"
#teacher_name="473714f244"

# 512-g-function:
#teacher_name="593ba334b9"


# Order of samples array has to correspond to order of dataset-array!
declare -a samples_array=()

#samples_array+=(100)
#samples_array+=(500)
#samples_array+=(1000)
#samples_array+=(10000)
#samples_array+=(30000)

samples_array+=(60000)
samples_array+=(180000)
samples_array+=(180000)
samples_array+=(360000)


#samples_array+=(1000)
#samples_array+=(1000)
#samples_array+=(1000)

#samples_array+=(180000)
#samples_array+=(180000)
#samples_array+=(180000)
#samples_array+=(180000)
#samples_array+=(180000)

#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)


#20240522 generation
#samples_array+=(5000)
#samples_array+=(50000)
#samples_array+=(50000)
#samples_array+=(50000)

#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)

#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(180000)
#samples_array+=(180000)
#samples_array+=(360000)

#samples_array+=(15000)
#samples_array+=(30000)
#samples_array+=(60000)
#samples_array+=(180000)

#samples_array+=(180000)
#samples_array+=(180000)

#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(360000)

#samples_array+=(5000)
#samples_array+=(90000)
#samples_array+=(180000)
#samples_array+=(360000)
#samples_array+=(360000)
#samples_array+=(180000)
#samples_array+=(5000)
#samples_array+=(360000)
#samples_array+=(360000)

# Steps = (#EP * |D|) / |Batch|
# |D| = samples_array
# |Batch| = 512 (by default)
# #EP = What we actually want to calculate (training_steps_array)
# Steps = We generally do not want more than 6 * 10^5
declare -a training_steps_array=()

#training_steps_array+=(1706)
#training_steps_array+=(1706)

training_steps_array+=(15000)
training_steps_array+=(15000)
training_steps_array+=(15000)
training_steps_array+=(15000)

#training_steps_array+=(9000)
#training_steps_array+=(9000)
#training_steps_array+=(9000)

#training_steps_array+=(6827)
#training_steps_array+=(6827)
#training_steps_array+=(6827)
#training_steps_array+=(6827)

#training_steps_array+=(1572000)
#training_steps_array+=(314400)
#training_steps_array+=(157200)
#training_steps_array+=(15720)
#training_steps_array+=(7360)

#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#20240522
#training_steps_array+=(50000)
#training_steps_array+=(5000)
#training_steps_array+=(5000)
#training_steps_array+=(5000)

#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)

#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(1706)
#training_steps_array+=(1706)
#training_steps_array+=(853)



#training_steps_array+=(25000)
#training_steps_array+=(25000)
#training_steps_array+=(1706)
#training_steps_array+=(853)
#training_steps_array+=(1706)
#training_steps_array+=(1706)
#training_steps_array+=(15000)
#training_steps_array+=(10000)
#training_steps_array+=(3413)
#training_steps_array+=(1706)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(853)
#
#training_steps_array+=(25000)
#training_steps_array+=(853)
#training_steps_array+=(853)
#training_steps_array+=(1706)
#training_steps_array+=(25000)

declare -a dataset_array=()

dataset_array+=("mnist")
dataset_array+=("mnist+mnist_random_noise_overlay_0_1_60k+mnist_random_noise_overlay_m1_0_60k")
dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3")
dataset_array+=("mnist_composition_sampling_360k_0d360_x3_y3")

#dataset_array+=("mnist+mnist_random_noise_overlay_m1_1_120k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_10_60k+mnist_random_noise_overlay_m10_0_60k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_1_60k+mnist_random_noise_overlay_m1_0_60k")

#dataset_array+=("mnist+mnist_random_noise_overlay_0_0p5_60k+mnist_random_noise_overlay_m0p5_0_60k")
#dataset_array+=("mnist+mnist_random_noise_overlay_m1_1_120k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_2_60k+mnist_random_noise_overlay_m2_0_60k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0p5_offsets_60k+mnist_random_noise_overlay_m0p5_offsets_60k")

#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_test")

#dataset_array+=("mnist+mnist_random_noise_overlay_0_1_60k+mnist_random_noise_overlay_m1_0_60k")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0")
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m1_0_87p5k+mnist_5k_random_noise_overlay_0_1_87p5k")
#dataset_array+=("mnist_composition_sampling_180k_0d0_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_random_noise_180k_min_0_max_1+mnist_random_noise_180k_min_m1_max_0")
#dataset_array+=("composition_sampling_random_noise_overlay_only_overlay_180k_0_1+composition_sampling_random_noise_overlay_only_overlay_180k_m1_0")
#dataset_array+=("mnist_random_noise_overlay_0_1_180k+mnist_random_noise_overlay_m1_0_180k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")

#20240522
#dataset_array+=("mnist_reduced_5k")
#dataset_array+=("mnist_random_noise_overlay_0_1_60k")
#dataset_array+=("mnist_random_noise_overlay_m1_0_60k")
#dataset_array+=("mnist_random_noise_overlay_m1_1_60k")

#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3")
#dataset_array+=("mnist_composition_sampling_360k_0d360_x3_y3")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x2_y1")
#dataset_array+=("mnist_composition_sampling_360k_0d360_x2_y1")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x4_y4")
#dataset_array+=("mnist_composition_sampling_360k_0d360_x4_y4")
#dataset_array+=("mnist_composition_sampling_180k_0d0_x3_y3")
#dataset_array+=("mnist_composition_sampling_360k_0d0_x3_y3")

#dataset_array+=("mnist+mnist_random_noise_overlay_0_1_150k+mnist_random_noise_overlay_m1_0_150k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_2_150k+mnist_random_noise_overlay_m2_0_150k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_0p5_150k+mnist_random_noise_overlay_m0p5_0_150k")
#dataset_array+=("mnist+mnist_random_noise_overlay_m1_1_300k")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_0p5_m0p5_0")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_2_m2_0")
#dataset_array+=("composition_sampling_random_noise_overlay_only_overlay_360k_m1_1")
#
#
#dataset_array+=("")
#
#
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_0_2_m2_0")
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_0_0p5_m0p5_0")

# ABLATION STUDY:
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m0p5_0_177p5k+mnist_5k_random_noise_overlay_0_0p5_177p5k")
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m1_0_177p5k+mnist_5k_random_noise_overlay_0_1_177p5k")
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m2_0_177p5k+mnist_5k_random_noise_overlay_0_2_177p5k")
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m1_1_355k")
#dataset_array+=("mnist_reduced_5k+mnist_5k_random_noise_overlay_m2_2_355k")

# ADDITIONAL 5k DATA:
#dataset_array+=("mnist_reduced_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_5k_5k_0_1_m1_0_reduced_mnist_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_10k_10k_0_1_m1_0_reduced_mnist_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_30k_30k_0_1_m1_0_reduced_mnist_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0_reduced_mnist_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_0_1_m1_0_reduced_mnist_5k")
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_m1_1_m1_1_reduced_mnist_5k")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3_reduced_mnist_5k")
#dataset_array+=("mnist_horizontal_flip_reduced_mnist_5k")
#dataset_array+=("")

#dataset_array+=("mnist_invert")
#dataset_array+=("mnist_vertical_flip")
#dataset_array+=("mnist_horizontal_flip_random_rotation")

#dataset_array+=("mnist+mnist_random_noise_overlay_m1_1_120k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_0p5_60k+mnist_random_noise_overlay_m0p5_0_60k")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_2_60k+mnist_random_noise_overlay_m2_0_60k")

#dataset_array+=("mnist_random_noise_overlay_m1_1_360k")

#dataset_array+=("mnist_random_noise_130k_min_0_max_1+mnist_random_noise_130k_min_m1_max_0")
#dataset_array+=("mnist_random_noise_360k_min_m1_max_1")

declare -i hour_index=0
for teacher_name in "${teacher_array[@]}"; do
    declare -i index=0
    minute="00"
    student_hidden_layer_size=${student_hidden_layer_array[hour_index]}
    for dataset in "${dataset_array[@]}"; do
        time="${hour}${minute}"
        echo "${date}-${time}-${dataset}"

        file_name="${date}_${time}.txt"
        cp ${template_file} ${file_name}

        sed -i -e "s/<REPLACE-NUMBER-STUDENTS>/${number_students}/g" ${file_name}
        sed -i -e "s/<REPLACE-HIDDEN-LAYER-SIZE>/${student_hidden_layer_size}/g" ${file_name}
        sed -i -e "s/<REPLACE-DATAGEN>/${dataset}/g" ${file_name}
        sed -i -e "s/<REPLACE-TEACHER-NAME>/${teacher_name}/g" ${file_name}
        sed -i -e "s/<REPLACE-SAMPLES>/${samples_array[index]}/g" ${file_name}
        sed -i -e "s/<REPLACE-STEPS>/${training_steps_array[index]}/g" ${file_name}
        sed -i -e "s/<REPLACE-TEACHER-SEED>/${teacher_seed}/g" ${file_name}
        sed -i -e "s/<REPLACE-GLOBAL-SEED>/${global_seed}/g" ${file_name}


        wandb_name="${date}-${time}-${dataset}"
        sed -i -e "s/<REPLACE-WANDB-NAME>/${wandb_name}/g" ${file_name}

        mkdir -p ${date}
        mv ${file_name} ${date}

        index+=1
        NUM_DIGITS=${#minute}
        minute=$(printf "%0${NUM_DIGITS}d" $((10#$minute + 1)))
    done
    hour_index=hour_index+1
    NUM_DIGITS=${#hour}
    hour=$(printf "%0${NUM_DIGITS}d" $((10#$hour + 1)))
done


#-------------------------------------------------
#Just a copy-paste list of possible teacher names:
#-------------------------------------------------
#
#Fulll MNIST:
#   g-32-Teacher: "e9b66ad22b"
#   g-64-Teacher: "a87448e112"
#   g-128-Teacher:"a260daf501"
#   g-256-Teacher:"1348026a3c"
#   g-512-teacher:"593ba334b9"
#5k-MNIST:
#   g-4-teacher: "09438e1695"
#   g-8-teacher: "6aca1e87a7"
#   g-16-teacher: "531cde1a25"
#   g-32-teacher: "70fbf466d8"
#   g-64-teacher: "b4b9a61d8a"
#   g-128-teacher: "6d174a27e7"
#   g-256-teacher: "10c5cd6563"
#   g-512-teacher: "473714f244"
#
#-------------------------------------------------
#Just a copy-paste list of possible datasets:
#-------------------------------------------------
#
#dataset_array+=("mnist_composition_sampling_360k_0d360_x3_y3")
#dataset_array+=("mnist_composition_sampling_180k_0d360_x3_y3")
#dataset_array+=("mnist_composition_sampling_60k_0d360_x3_y3")
#dataset_array+=("mnist")
#dataset_array+=("mnist_random_rotation")
#dataset_array+=("mnist_horizontal_flip")
#dataset_array+=("mnist_composition_sampling_120k_0d360_x3_y3")
#dataset_array+=("mnist+mnist_random_noise_overlay_0_1_60k+mnist_random_noise_overlay_m1_0_60k")
#dataset_array+=("composition_sampling_random_noise_overlay_30k_30k_0_1_m1_0")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_0_1_m1_0")
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_0_1_m1_0")
#dataset_array+=("composition_sampling_random_noise_overlay_multiplication_120k_120k_1_2_m2_m1")
#dataset_array+=("composition_sampling_random_noise_overlay_60k_60k_m1_1_m1_1")
#dataset_array+=("composition_sampling_random_noise_overlay_120k_120k_m1_1_m1_1")


