import concurrent.futures
import hashlib
import os
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import stats
from tqdm import tqdm

from src.data_provider.bin_data_reader import BinDataReader
from src.file_logger import FileLogger
from src.handlers import _Handler
from src.parsers.enums import DataFormatType
from src.parsers.interfaces import _Args
from src.path_helper import PathHelper


class HandlerModelFeatureStats(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.sequence_index = args.sequence_index
            self.data_format = args.data_format
            self.feature_dims = args.feature_dims
            self.coord_dims = args.coord_dims
            self.live_plot = args.live_plot
            self.threads = args.threads

    def handle(self):
        args: HandlerModelFeatureStats.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            self.handle_kitti(args)
            return

        raise NotImplementedError()

    def handle_kitti(self, args):
        file_log = FileLogger()
        path_helper = PathHelper().generated().dataset(args.dataset_index).sequences()
        bins_path = path_helper.sequence(args.sequence_index).bins().path()
        bin_names = os.listdir(bins_path)
        bin_names.sort()

        bin_datas = {}

        # Reading is fast.
        progress = tqdm(desc=f"Reading data   ", total=len(bin_names))
        for bin_name in bin_names:
            bin_path = os.path.join(bins_path, bin_name)
            bin_data = BinDataReader.read_bin_data(
                bin_path, feature_dims=args.feature_dims, coord_dims=args.coord_dims)
            bin_datas[bin_name] = bin_data
            progress.update(1)
        progress.close()

        thread_count = args.threads

        call_args = []
        feature_data = np.array([], dtype=np.float32)

        # Appending is slow so we thread.
        for thread_index in range(thread_count):
            call_args.append((
                np.array([], dtype=np.float32),
                bin_names[thread_index::thread_count],
                bin_datas, args.coord_dims))

        progress = tqdm(desc=f"Combining data ", total=len(bin_datas))
        if thread_count == 1:
            # For ease of debugging.
            feature_data = self.append_data(*call_args[0], progress)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(self.append_data, *call_args_, progress) for call_args_ in call_args]
                for future in concurrent.futures.as_completed(futures):
                    feature_data = np.append(feature_data, future.result())
        progress.close()

        for active_feature_dim in range(0, args.feature_dims):
            file_log.log(f"Summarising feature_dim ({active_feature_dim + 1}/{args.feature_dims})...")
            self.summarise_array(active_feature_dim, feature_data, show_plots=args.live_plot)

    @staticmethod
    def append_data(to_data, keys_to_append, data_map, coord_dims, progress):
        for key in keys_to_append:
            # All rows. But skip coord_dims, keep only feature_dims.
            to_data = np.append(to_data, data_map[key][:, coord_dims:])
            if progress:
                progress.update(1)
        return to_data

    @staticmethod
    def summarise_array(feature_dim, arr, show_plots=True):
        if arr.ndim != 1:
            raise ValueError("Input must be a 1D NumPy array.")

        arr = arr[~np.isnan(arr)]  # remove NaNs if any

        # Basic stats.
        count = len(arr)
        mean = np.mean(arr)
        median = np.median(arr)
        std = np.std(arr)
        minimum = np.min(arr)
        maximum = np.max(arr)
        data_range = np.ptp(arr)
        q1 = np.percentile(arr, 25)
        q2 = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        skewness = stats.skew(arr)
        kurt = stats.kurtosis(arr)

        # Outlier detection.
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = arr[(arr < lower_bound) | (arr > upper_bound)]

        # Display results.
        results_string = f"""
----------------------------------
  Summary (feature_dim={feature_dim}):
  
    Count     : {count}
    Mean      : {mean:.3f}
    Median    : {median:.3f}
    Std Dev   : {std:.3f}
    Min       : {minimum:.3f}
    Max       : {maximum:.3f}
    Range     : {data_range:.3f}
    Q1        : {q1:.3f}
    Q2 (Med)  : {q2:.3f}
    Q3        : {q3:.3f}
    IQR       : {iqr:.3f}
    Skewness  : {skewness:.3f}
    Kurtosis  : {kurt:.3f}
    Outliers  : {len(outliers)}
"""

        print(results_string)
        results_sha = hashlib.sha256(results_string.encode())
        print(f"    Results SHA: {results_sha.hexdigest()[:8]}")
        print()

        # Plots.
        if show_plots:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            axs[0].hist(arr, bins=20, color='skyblue', edgecolor='black')
            axs[0].set_title("Histogram")

            axs[1].boxplot(arr, vert=False)
            axs[1].set_title("Boxplot")

            plt.tight_layout()
            plt.show()
