import cv2
import rioxarray as rxr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt


# Read NAIP data from file into xarray.DataArray
img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
naip_img = rxr.open_rasterio(img_path)

print("NAIP image shape: ", naip_img.shape)

# Convert xarray.DataArray to numpy array
naip_img = naip_img.to_numpy()

# Cast uint8 to float64 to prevent calculation from overflowing
naip_data = naip_img.astype(np.float64)

# Calculate NDVI
naip_ndvi = es.normalized_diff(naip_data[3], naip_data[0])

# Set classification threshold
threshold = 0.12

# Label green space based on threshold
naip_ndvi[naip_ndvi >= threshold] = 1
naip_ndvi[naip_ndvi < threshold] = 0

print("NAIP NDVI shape:", naip_ndvi.shape)
print("NAIP NDVI dtype:", naip_ndvi.dtype)
# Visualize classification results
# ep.plot_bands(naip_ndvi,
#               cmap='PiYG',
#               scale=False,
#               vmin=-1, vmax=1,
#               title="NDVI")

# ep.hist(naip_ndvi,
#         figsize=(12, 6),
#         title=["NDVI: Distribution of pixels\n NAIP 2015 Cold Springs fire site"])

# plt.savefig("./image/ndvi/ndvi_01.png", dpi=300, format="png")
# plt.show()

cc_input = naip_ndvi.astype(np.uint8)
cc_output = cv2.connectedComponentsWithStats(cc_input, 8, cv2.CV_32S)
(num_labels, labels, stats, centroids) = cc_output

for i in range(num_labels):
    if i == 0:
        text = f"Examing component {i + 1}/{num_labels} (background)"
    else:
        text = f"Examing component {i + 1}/{num_labels}"
    print(f"[INFO] {text}")

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (c_x, c_y) = centroids[i]

    cc_output = naip_ndvi.copy()
    cv2.rectangle(cc_output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(cc_output, (int(c_x), int(c_y)), 4, (0, 0, 255), -1)

    component_mask = (labels == i).astype("uint8") * 255
    cv2.imshow("Output", cc_output)
    cv2.imshow("Connected Component", component_mask)
    cv2.waitKey(0)