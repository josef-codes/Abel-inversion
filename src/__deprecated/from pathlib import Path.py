import os

def get_batch_names(root_folder):
    return [
        name for name in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, name))
    ]


# ── Usage ──
root = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25"
batches = get_batch_names(root)
print(batches)