ORI_DATA_ROOT="/home/jensen/remote_jensen/huangjianxin/detany3d_dataset/datasets"
OUT_DATA_ROOT="/home/jensen/remote_jensen/huangjianxin/vlm_benchmark/Omni3D"

python export_annotations.py \
--json $ORI_DATA_ROOT/Omni3D/SUNRGBD_test.json \
--data_root $ORI_DATA_ROOT \
--output_dir $OUT_DATA_ROOT/SUNRGBD

python export_annotations.py \
--json $ORI_DATA_ROOT/Omni3D/ARKitScenes_test.json \
--data_root $ORI_DATA_ROOT \
--output_dir $OUT_DATA_ROOT/arkitscenes

python export_annotations.py \
--json $ORI_DATA_ROOT/Omni3D/Hypersim_test.json \
--data_root $ORI_DATA_ROOT \
--output_dir $OUT_DATA_ROOT/hypersim

python export_annotations.py \
--json $ORI_DATA_ROOT/Omni3D/Objectron_test.json \
--data_root $ORI_DATA_ROOT \
--output_dir $OUT_DATA_ROOT/objectron