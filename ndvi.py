import cv2
import rioxarray as rxr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Read NAIP data from file into xarray.DataArray
img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
naip_img = rxr.open_rasterio(img_path)

print("NAIP image shape: ", naip_img.shape)

# Convert xarray.DataArray to numpy array
naip_img = naip_img.to_numpy()

# Extract RGB channels and reorder the color axis from (c, w, h) to (w, h, c)
naip_rgb = np.moveaxis(naip_img[0:3], 0, -1)
print("NAIP RGB image shape:", naip_rgb.shape)


# Convert RGB to BGR, as BGR is the default color model of OpenCV
naip_rgb = cv2.cvtColor(naip_rgb, cv2.COLOR_RGB2BGR)
# cv2.imwrite("./image/naip_rgb.png", naip_rgb)
# cv2.imshow("NAIP RGB Image", naip_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Cast uint8 to float64 to prevent calculation from overflowing
naip_data = naip_img.astype(np.float64)

# Calculate NDVI. Input must be numpy array. xarray does not support 2D boolean indexing
naip_ndvi = es.normalized_diff(naip_data[3], naip_data[0])

# Set classification threshold
threshold = 0.2

# Label green space based on threshold
naip_ndvi[naip_ndvi >= threshold] = 1
naip_ndvi[naip_ndvi < threshold] = 0

print("NAIP NDVI shape:", naip_ndvi.shape)
print("NAIP NDVI dtype:", naip_ndvi.dtype)

# Visualize classification results
ep.plot_bands(naip_ndvi,
              cmap='PiYG',
              scale=False,
              vmin=-1, vmax=1,
              title="NDVI")

plt.savefig("./image/ndvi/ndvi_map.png", dpi=300, format="png")
plt.show()


# ep.hist(naip_ndvi,
#         figsize=(12, 6),
#         title=["NDVI: Distribution of pixels\n NAIP 2015 Cold Springs fire site"])

# plt.savefig("./image/ndvi/ndvi_01.png", dpi=300, format="png")
# plt.show()

# Convert dtype to uint8
cc_input = naip_ndvi.astype(np.uint8)

# Input cv2.connectedComponentsWithStats() must be 8-bit single-channel image
connectivity = 8  # choose 4-way or 8-way connectivity
cc_output = cv2.connectedComponentsWithStats(cc_input, connectivity, cv2.CV_32S)
(num_labels, labels, stats, centroids) = cc_output
print(f"Conectivity = {connectivity}")
print(f"Shape of Statistics: {stats.shape}")
print(f"Shape of labels: {labels.shape}")
print(f"Shape of centroids: {centroids.shape}")

# Calculate summary statistics of area
area_df = pd.DataFrame(stats[:, [cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA]],
                       columns=['Width', 'Height', 'Area'])
area_stats = area_df.describe()
print(area_stats.round(2))

# Merge connected components information
cc_info = np.hstack((stats, centroids))
cc_info_df = pd.DataFrame(cc_info, columns=['LeftMost', 'TopMost', 'Width',
                                            'Height', 'Area', 'CentroidsX', 'CentroidsY'])
# Sort by area
cc_info_df_sorted = cc_info_df.sort_values(by=['Area'])

# Filter out based on width and height
cc_info_df_filtered = cc_info_df_sorted.loc[(cc_info_df_sorted['Width'] > 10) &
                                            (cc_info_df_sorted['Height'] > 10) &
                                            (cc_info_df_sorted['LeftMost'] > 0) &
                                            (cc_info_df_sorted['TopMost'] > 0)]

# Save to file
cc_info_df_sorted.to_csv("connected_components_info_all.csv", index=False)
cc_info_df_filtered.to_csv("connected_components_info_filtered.csv", index=False)

keep_count = 0
video_frames = []
size = (0, 0)
for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (c_x, c_y) = centroids[i]

    keep_w = w > 10
    keep_h = h > 10

    if all((keep_w, keep_h)):
        keep_count += 1
        print(f"[INFO] keeping connected component {i}")

        cc_output = naip_rgb.copy()
        cv2.rectangle(cc_output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(cc_output, (int(c_x), int(c_y)), 4, (0, 0, 255), -1)

        component_mask = (labels == i).astype("uint8") * 255
        component_mask_rgb = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
        video_frame = np.hstack((component_mask_rgb, cc_output))
        # size = (video_frame.shape[1], video_frame.shape[0])
        # print(video_frame.shape)
        video_frames.append(video_frame)
        # cv2.imshow("Output", cc_output)
        # cv2.imshow("Connected Component", component_mask)
        # cv2.waitKey(0)
print(f"Kept {keep_count}/{num_labels} connected components")
# cv2.destroyAllWindows()

if len(video_frames) > 0:
    size = (video_frames[0].shape[1], video_frames[0].shape[0])
    video_out = cv2.VideoWriter('cc_video01.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1, size)

    for i in range(len(video_frames)):
        video_out.write(video_frames[i])
    video_out.release()


