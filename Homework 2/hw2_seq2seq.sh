
#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./hw2_seq2seq.sh <data_directory> <test_data_output_filename>"
    exit 1
fi

# Assign arguments to variables
DATA_DIR=$1
OUTPUT_FILE=$2

# Run your Python script with the provided arguments
python /content/gdrive/MyDrive/hw2_hw2_1/SeqtoSeqTrain.py $DATA_DIR $OUTPUT_FILE
python /content/gdrive/MyDrive/hw2_hw2_1/bleu_eval.py /content/gdrive/MyDrive/hw2_hw2_1/result.txt /content/gdrive/MyDrive/hw2_hw2_1/testing_data/testing_label.json


