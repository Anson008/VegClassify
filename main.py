from filter_factory import FilterFactory
from connected_components import CV2ConnectedComponentsGenerator, ConnectedComponents
from naip_processor import NAIPProcessor

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img_path = "./image/m_4111118_nw_12_060_20210813_Clip.tif"
    naip = NAIPProcessor(img_path)
    naip_rgb = naip.get_rgb_naip()
    naip_reprojected = naip.reproject("EPSG:4326")
    ndvi = NAIPProcessor.calculate_ndvi(naip_reprojected)
    ndvi_classified = NAIPProcessor.classify(ndvi, 0.11)
    cv2_cc_generator = CV2ConnectedComponentsGenerator(ndvi_classified, 8)
    cc_results = cv2_cc_generator.generate()

    cc_object = ConnectedComponents(cc_results)
    # area_stats = cc_object.summary_statistics()
    # print(area_stats.round(2))

    filter_factory = FilterFactory()
    filters = []
    filters.append(filter_factory.get_filter("height", "<=", 50))
    filters.append(filter_factory.get_filter("width", ">", 20))
    cc_object_filtered = cc_object.apply_filters(filters)
    print(cc_object_filtered.summary_statistics())

