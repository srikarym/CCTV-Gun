_base_ = ["./default_runtime.py"]

# save best model
total_epochs = 36
evaluation = dict(
    metric = "bbox",
    interval = 1,
    save_best = 'bbox_h_mAP',
    rule = "greater"
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 8.
auto_scale_lr = dict(enable=True, base_batch_size=8)

workflow = [('train', 1), ('val', 1)]

# only save final model
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

