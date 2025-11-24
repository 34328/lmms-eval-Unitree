export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.3.0.86:8888/v1"
export OPENAI_API_URL="http://10.3.0.86:8888/v1"
export DATA_ROOT="/home/jensen/remote_jensen/huangjianxin/detany3d_dataset/datasets/"
export SAVE_VISUAL_PATH="./visual_results/"

python -m lmms_eval  --model async_openai  --model_args model_version="Qwen3-VL-235B-A22B",fps=1  --tasks omni3d_sunrgbd --batch_size 1  --log_samples  --output_path ./logs/ --limit 10 --verbosity=DEBUG