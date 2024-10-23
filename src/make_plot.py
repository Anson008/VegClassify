import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

if __name__ == "__main__":
    ndvi_data_path = "../cache/ndvi_thresholds_on_accuracy.npy"
    ndvi_data = np.load(ndvi_data_path)
    print(ndvi_data.shape)

    ndvi_data = ndvi_data.T

    fig, axe = plt.subplots()
    axe.plot(ndvi_data[0], ndvi_data[1], marker=".", markersize=12)
    axe.set_xlim([-0.02, 0.4])
    axe.set_ylim([0.8, 0.9])
    axe.xaxis.set_minor_locator(AutoMinorLocator())
    axe.yaxis.set_minor_locator(AutoMinorLocator())
    axe.tick_params(axis="both", which="both", direction="in")

    axe.set_xlabel("NDVI Threshold", fontsize=14)
    axe.set_ylabel("Accuracy", fontsize=14)

    # output_path = "./results/NDVI_Thresholds/ndvi_kappa.png"
    output_path = "../results/NDVI_Threshold/ndvi_accuracy.png"
    plt.savefig(fname=output_path, dpi=300, format="png")
    plt.show()
