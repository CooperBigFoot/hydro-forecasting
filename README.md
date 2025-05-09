## Useful when designing experiments

Helper Function for Checkpoint Filename: While _find_checkpoint_in_attempt_dir assumes one .ckpt file, if Pytorch Lightning sometimes leaves other files (e.g., last.ckpt even with save_top_k=1 for the best), the logic might need to be more specific in selecting the "best" one if multiple are present, or rely strictly on the guidleine that ModelCheckpoint is configured to only leave one. Usually, trainer.checkpoint_callback.best_model_path is the source of truth for the best checkpoint from a training run.
