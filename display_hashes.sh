# Small script for creating configuration files.
# Replace (starting)time, date, number_students, etc. programmatically.

output_file="20240813_deep_teachers.csv"
#folder="exps/shallow_mnist/ec_g/"
#folder="exps/shallow_visualizations_test/ec_g/"
folder="exps/deep_mnist/g/"
date="20240813"
declare -i index=0

# Order of samples array has to correspond to order of dataset-array!
declare -a file_array=()

#file_array+=("20240515/20240515_1000.txt")
for hour in {10..13}; do
    for minute in {00..00}; do
        file_array+=("${date}/${date}_${hour}${minute}.txt")
    done
done

# Emtying output file:
#echo "" > ${output_file}

for file in "${file_array[@]}"; do
    echo "${file}"
    file_path=${folder}${file}

    if [ -e "$file_path" ]; then
        my_hash=$(python EC.py $(cat ${file_path}) --display_hashname)
        
        if [ "$index" -eq 0 ]; then
            echo "${my_hash}" > ${output_file}
        else
            echo "${my_hash}" >> ${output_file}
        fi

        index+=1
    fi
done


#-------------------------------------------------
#Just a copy-paste list of possible teacher names:
#-------------------------------------------------
#
#folder="exps/shallow_mnist/g/"
#
#
#
