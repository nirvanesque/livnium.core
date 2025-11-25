#!/bin/bash

echo "============================================================"
echo "            PHASE 4: NON-LINEAR DECODER PIPELINE           "
echo "============================================================"

set -e

# Move into PHASE 4 directory
cd "$(dirname "$0")"

echo ""
echo "STEP 1: Training Decoder"
python3 code/fit_center_decoder.py \
    --n-components 8 \
    --output-dir results \
    --verbose

echo ""
echo "STEP 2: Running Shadow Rule 30 (with Non-Linear Decoder)"
python3 code/shadow_rule30_phase4.py \
    --data-dir ../PHASE3/results \
    --decoder-dir results \
    --output-dir results \
    --num-steps 5000 \
    --verbose

echo ""
echo "============================================================"
echo "PHASE 4 COMPLETE!"
echo "Decoder + Shadow Model Output Saved In: results/"
echo "============================================================"
