# Small script for creating configuration files (training).
# Replace (starting)time, date, number_students, etc. programmatically.
declare -i time=1000
date=20240515
declare -a teacher_size_array=()
teacher_size_array+=(4)
teacher_size_array+=(8)
teacher_size_array+=(16)
teacher_size_array+=(32)
teacher_size_array+=(64)
teacher_size_array+=(128)
teacher_size_array+=(256)
teacher_size_array+=(512)

template_file="configuration_template_train.txt"

global_seed=1
dataset="mnist_reduced_5k"
activation_function="g"

for teacher_size in "${teacher_size_array[@]}"; do
    echo "${date}-${time} -> Teacher-Size:${teacher_size}"

    file_name="${date}_${time}.txt"
    cp ${template_file} ${file_name}

    sed -i -e "s/<REPLACE-TEACHER-SIZE>/${teacher_size}/g" ${file_name}
    sed -i -e "s/<REPLACE-GLOB-SEED>/${global_seed}/g" ${file_name}
    sed -i -e "s/<REPLACE-ACT-FUN>/${activation_function}/g" ${file_name}
    sed -i -e "s/<REPLACE-DATASET-NAME>/${dataset}/g" ${file_name}

    mkdir -p ${date}
    mv ${file_name} ${date}

    time=time+1
done


#-------------------------------------------------
#Just a copy-paste list of possible datasets:
#-------------------------------------------------
#
#dataset="mnist_reduced_5k"
#dataset="mnist_reduced_1k"
#dataset="mnist"

