#!/bin/bash

# Define the number of threads to test
THREAD_COUNTS=(1 2 4 8 16 32)
REPEAT_COUNT=5
OUTPUT_FILE="timelog.csv"

# Create the header for the CSV file
echo "Thread Count,Run,Time Taken (s)" > $OUTPUT_FILE

# Loop over each thread count
for THREAD_COUNT in "${THREAD_COUNTS[@]}"; do
    # Repeat the test for each thread count
    for RUN in $(seq 1 $REPEAT_COUNT); do
        # Set the thread count
        # export OMP_NUM_THREADS=$THREAD_COUNT

        # Measure the time taken for each run
        START_TIME=$(date +%s.%N)
        
        # Execute the program and output to image.ppm
        ./src/InOneWeekend/main 1920 $THREAD_COUNT $THREAD_COUNT >> image.ppm
        
        END_TIME=$(date +%s.%N)

        # Calculate the elapsed time
        TIME_TAKEN=$(echo "$END_TIME - $START_TIME" | bc)

        # Log the results in the CSV file
        echo "$THREAD_COUNT,$RUN,$TIME_TAKEN" >> $OUTPUT_FILE
    done
done

echo "Timing test completed. Results are saved in $OUTPUT_FILE."
