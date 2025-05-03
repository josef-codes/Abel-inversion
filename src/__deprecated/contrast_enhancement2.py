
import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt

# 1) Load your 16‑bit image
path = 'C:\\Users\\User\\z\\Desktop\\WUT\\Diplomka\\DATA\\Images\\H0_3_28_25\\50-1000ns\\M3_X8.tif'
img16 = io.imread(path).astype(np.uint16)

# 2) Normalize to [0,1]
max_val = np.iinfo(img16.dtype).max
img_norm = img16.astype(np.float64) / max_val

# 3) Apply gamma correction (e.g. γ = 0.5 to brighten)
gamma = 0.5
img_gamma = exposure.adjust_gamma(img_norm, gamma=gamma)

# 4) Convert back to 16‑bit
img_gamma16 = (img_gamma * max_val).astype(np.uint16)



# 5) Display before and after
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img16, cmap='gray', vmin=0, vmax=max_val)
axes[0].set_title('Original (16‑bit)')
axes[0].axis('off')

axes[1].imshow(img_gamma16, cmap='gray', vmin=0, vmax=max_val)
axes[1].set_title(f'Gamma corrected (γ={gamma})')
axes[1].axis('off')

plt.tight_layout()
plt.show()
