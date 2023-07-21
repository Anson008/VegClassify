import rioxarray as rxr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import numpy as np
import matplotlib.pyplot as plt


img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
naip_data = rxr.open_rasterio(img_path)

print("Image shape: ", naip_data.shape)
naip_data = naip_data.to_numpy()
naip_data = naip_data.astype(np.float64)
print(type(naip_data))
print(naip_data[3].shape)

naip_ndvi = es.normalized_diff(naip_data[3], naip_data[0])

threshold = 0.15
naip_ndvi[naip_ndvi >= threshold] = 1
naip_ndvi[naip_ndvi < threshold] = 0

ep.plot_bands(naip_ndvi,
              cmap='PiYG',
              scale=False,
              vmin=-1, vmax=1,
              title="NDVI")

# ep.hist(naip_ndvi,
#         figsize=(12, 6),
#         title=["NDVI: Distribution of pixels\n NAIP 2015 Cold Springs fire site"])

plt.show()
