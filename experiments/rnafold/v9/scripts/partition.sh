#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <input_file> <output_file_selected> <output_file_remaining> <percentage>"
  exit 1
fi

input_file="$1"
output_file_selected="$2"
output_file_remaining="$3"
percentage="$4"

# Validate that percentage is a number and between 0 and 100
if ! [[ "$percentage" =~ ^[0-9]+$ ]] || [ "$percentage" -lt 0 ] || [ "$percentage" -gt 100 ]; then
  echo "Error: Percentage must be a number between 0 and 100."
  exit 1
fi

# Check if the input file exists
if [ ! -f "$input_file" ]; then
  echo "Error: Input file '$input_file' does not exist."
  exit 1
fi

# Count the total number of lines in the input file
total_lines=$(wc -l < "$input_file")

# Calculate the specified percentage of the total lines
lines_to_select=$(echo "scale=0; $total_lines * $percentage / 100" | bc)

# Shuffle the lines in the input file
shuffled_file=$(mktemp)
shuf "$input_file" > "$shuffled_file"

# Split the shuffled file into selected and remaining lines
head -n "$lines_to_select" "$shuffled_file" > "$output_file_selected"
tail -n +$((lines_to_select + 1)) "$shuffled_file" > "$output_file_remaining"

# Clean up temporary file
rm "$shuffled_file"

# Notify the user
echo "Randomly selected $percentage% of lines written to '$output_file_selected'."
echo "Remaining lines written to '$output_file_remaining'."
