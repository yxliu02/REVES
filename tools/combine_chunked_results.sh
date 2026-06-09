#!/bin/bash
# ============================
# Combine chunked math dataset results
# ============================

set -e  # Exit immediately if any command fails
set -u  # Error on use of undefined variables

# Usage: ./combine_chunked_results.sh <input_dir> <total_chunks> <chunk_pattern> <output_jsonl>
# Example: ./combine_chunked_results.sh ./output 4 "part3_world4" combined.jsonl

if [ $# -ne 4 ]; then
    echo "Usage: $0 <input_dir> <total_chunks> <chunk_pattern> <output_jsonl>"
    echo "Example: $0 ./output 4 \"part3_world4\" combined.jsonl"
    echo ""
    echo "Arguments:"
    echo "  input_dir     : Directory containing chunk result files"
    echo "  total_chunks  : Total number of chunks"
    echo "  chunk_pattern : Pattern for chunk files (e.g., 'part3_world4' for files like part3_world4_1.jsonl, part3_world4_2.jsonl)"
    echo "  output_jsonl  : Output combined JSONL file"
    exit 1
fi

INPUT_DIR=$1
TOTAL_CHUNKS=$2
CHUNK_PATTERN=$3
OUTPUT_JSONL=$4

echo "Combining chunked results..."
echo "Input directory: $INPUT_DIR"
echo "Total chunks: $TOTAL_CHUNKS"
echo "Chunk pattern: $CHUNK_PATTERN"
echo "Output JSONL: $OUTPUT_JSONL"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# Create output directory
OUTPUT_DIR=$(dirname "$OUTPUT_JSONL")
mkdir -p "$OUTPUT_DIR"

# Find all chunk files matching the pattern
CHUNK_FILES=()
for i in $(seq 1 $TOTAL_CHUNKS); do
    CHUNK_FILE="${INPUT_DIR}/${CHUNK_PATTERN}_${i}.jsonl"
    if [ -f "$CHUNK_FILE" ]; then
        CHUNK_FILES+=("$CHUNK_FILE")
        echo "Found chunk file: $CHUNK_FILE"
    else
        echo "Warning: Chunk file not found: $CHUNK_FILE"
    fi
done

if [ ${#CHUNK_FILES[@]} -eq 0 ]; then
    echo "Error: No chunk files found matching pattern ${CHUNK_PATTERN}_*.jsonl"
    exit 1
fi

echo "Found ${#CHUNK_FILES[@]} chunk files to combine"

# Combine JSONL files
echo "Combining JSONL files..."
cat "${CHUNK_FILES[@]}" > "$OUTPUT_JSONL"
echo "Combined JSONL saved to: $OUTPUT_JSONL"

# Count lines in combined file
TOTAL_LINES=$(wc -l < "$OUTPUT_JSONL")
echo "Total lines in combined JSONL: $TOTAL_LINES"

echo "JSONL combination completed!"

# Show summary
echo ""
echo "=== Summary ==="
echo "Input chunks: ${#CHUNK_FILES[@]}"
for chunk_file in "${CHUNK_FILES[@]}"; do
    chunk_lines=$(wc -l < "$chunk_file")
    echo "  $(basename "$chunk_file"): $chunk_lines lines"
done
echo "Combined JSONL: $TOTAL_LINES lines"
echo "Output file:"
echo "  JSONL: $OUTPUT_JSONL"

echo ""
echo "Combination completed successfully!"
