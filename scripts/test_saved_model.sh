#!/bin/bash
# Test saved SNLI model

cd /Users/chetanpatil/Desktop/clean-nova-livnium
source .venv/bin/activate
export PYTHONPATH=/Users/chetanpatil/Desktop/clean-nova-livnium:$PYTHONPATH

# Check if model exists
if [ ! -f "logs/snli_model.pt" ]; then
    echo "Error: Model file not found at logs/snli_model.pt"
    exit 1
fi

# Check if test data exists
TEST_DATA="livnium/domains/snli/data/snli_1.0_test.jsonl"
if [ ! -f "$TEST_DATA" ]; then
    echo "Warning: Test data not found at $TEST_DATA"
    echo "Using dev set instead..."
    TEST_DATA="livnium/domains/snli/data/snli_1.0_dev.jsonl"
fi

echo "=========================================="
echo "Testing SNLI Model"
echo "=========================================="
echo "Model: logs/snli_model.pt"
echo "Test data: $TEST_DATA"
echo ""

python3 livnium/examples/test_snli.py \
    --checkpoint logs/snli_model.pt \
    --test "$TEST_DATA" \
    --dim 256 \
    --layers 5 \
    --vocab-size 2000 \
    --batch-size 64

echo ""
echo "=========================================="
echo "Test complete"
echo "=========================================="

