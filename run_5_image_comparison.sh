# python3 infer_sam.py \
#   --config configs/full_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/eaton_on_2025_01_10_2688_16128.png \
#   --prompt "building with no damage" \
#   --output predictions.png


# python3 infer_sam.py \
#   --config configs/full_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/eaton_on_2025_01_10_2688_16128.png \
#   --prompt "building with no damage" \
#   --output predictions.png


# python3 infer_sam.py \
#   --config configs/full_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/palisades_post_2025_07_12_65408_36736.png \
#   --prompt "debris_cleared" \
#   --output predictions.png



# python3 infer_sam.py \
#   --config configs/full_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/palisades_post_2025_07_12_65408_36736.png \
#   --prompt "building with no damage" \
#   --output predictions.png



# python3 infer_sam.py \
#   --config configs/light_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/palisades_post_2025_07_12_65408_36736.png \
#   --prompt "building with no damage" \
#   --output predictions.png

# python3 infer_sam.py \
#   --config configs/light_lora_config.yaml \
#   --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/palisades_post_2025_07_12_65408_36736.png \
#   --prompt "debris_cleared" \
#   --output predictions.png


python3 infer_sam.py \
  --config configs/light_lora_config.yaml \
  --image ../sam3/data/eaton_palisades_30cm_4snap_temporal_refined_v1/test/images/palisades_very_post_2025_10_28_60928_34048.png \
  --prompt "debris_cleared" \
  --output predictions.png