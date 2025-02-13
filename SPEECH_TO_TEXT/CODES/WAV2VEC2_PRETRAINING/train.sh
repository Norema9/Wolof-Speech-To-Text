CUDA_VISIBLE_DEVICES="0" accelerate launch \
	--multi_gpu \
	--num_machines="1" \
	--num_processes="2" \
	--mixed_precision="fp16" \
	--num_cpu_threads_per_process="12" \
	run.py \
		--train_datasets \ 
			E:\\IA\\WOLOF\\SPEECH_TO_TEXT\\DATA\\CLEANED\\WOLOF_AUDIO_TRANS\\validation_data.csv \
		--val_datasets \
			E:\\IA\\WOLOF\\SPEECH_TO_TEXT\\DATA\CLEANED\\WOLOF_AUDIO_TRANS\\test_data.csv \
		--audio_column_name="path" \
		--duration_column_name="duration" \
		--separator="," \
		--model_name_or_path="facebook/wav2vec2-base" \
		--load_from_pretrained \
		--output_dir="E:\\IA\\WOLOF\\SPEECH_TO_TEXT\\MODELS\\PRETRAINED_WAV2VEC2" \
		--max_train_steps="300000" \
		--num_warmup_steps="90000" \
		--gradient_accumulation_steps="8" \
		--learning_rate="0.005" \
		--weight_decay="0.01" \
		--max_duration_in_seconds="15.6" \
		--min_duration_in_seconds="0.5" \
		--logging_steps="1" \
		--saving_steps="100" \
		--per_device_train_batch_size="16" \
		--per_device_eval_batch_size="8" \
		--adam_beta1="0.9" \
		--adam_beta2="0.98" \
		--adam_epsilon="1e-06" \
		--gradient_checkpointing

