#!/bin/bash

# Initialize VARIABLE with leading zeros
VARIABLE="0075"
echo $VARIABLE

# Calculate the number of digits including leading zeros
NUM_DIGITS=${#VARIABLE}

# Increment the variable by removing leading zeros, incrementing, and formatting it back with leading zeros
# 10#$VARIABLE treats the variable as a decimal number (removing leading zeros for arithmetic operations)
# $(( ... )) performs the arithmetic operation
# printf "%0${NUM_DIGITS}d" formats the number with leading zeros to match the original length
VARIABLE=$(printf "%0${NUM_DIGITS}d" $((10#$VARIABLE + 1)))

# Print the result to verify
echo $VARIABLE
