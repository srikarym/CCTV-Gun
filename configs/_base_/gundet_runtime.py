_base_ = ["./default_runtime.py"]

total_epochs = 12
evaluation = dict(
    metric = "bbox",
    interval = 1
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = 8.
auto_scale_lr = dict(enable=True, base_batch_size=8)

workflow = [('train', 1), ('val', 1)]

checkpoint_config = dict(interval=1)

