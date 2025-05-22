import numpy as np
from sklearn.neighbors import KernelDensity
import scipy
import pandas as pd
from tqdm import tqdm
from config_management.data_config_muon_decay import DataConfig

seed = 42

def kde1D(x, bandwidth, xbins=100j, **kwargs):
    """Build 1D kernel density estimate (KDE)."""
    X = np.mgrid[x.min():x.max():xbins]
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x.reshape(-1, 1))
    density = np.exp(kde_skl.score_samples(X.reshape(-1, 1)))
    return X, density

def read_file(file_path, columns):
    print("reading file...")
    df = pd.read_csv(file_path)
    samples = df[columns].to_numpy()
    return samples

def generate_marginals(samples, num_var, save_prefix, b_adj=1e-2):
    print("generating marginals...")
    for vid in tqdm(range(num_var)):
        file_name = f"./data/processed_data/{save_prefix}_marginal_data_{vid}.csv"
        bw = b_adj * scipy.stats.gaussian_kde(samples[:, vid]).scotts_factor()
        X, density = kde1D(samples[:, vid], bw)
        data = np.hstack([X.reshape(-1, 1), density.reshape(-1, 1)])
        np.savetxt(file_name, data, delimiter=",")


def generate_conditionals(samples, save_prefix, K = 'gaussian', slice_num=8, b_adj=1e-2):
    # conditional_data_[0-1]_slice_[0-7].csv
    # conditional_slices_[0-1].csv
    print("generating conditionals...")

    bw = b_adj * scipy.stats.gaussian_kde(samples.T).scotts_factor()
    kde_all = KernelDensity(bandwidth=bw, kernel=K)
    kde_all.fit(samples)
    var_num = samples.shape[1]
    for vid in tqdm(range(var_num)):
        kde_x_other = KernelDensity(bandwidth=bw, kernel=K)
        data = np.zeros((samples.shape[0], var_num - 1))
        for i in range(var_num - 1):
            data[:, i] = samples[:, i + (i >= vid)]
        kde_x_other.fit(data)
        x_other_sample = kde_x_other.sample(slice_num, random_state=seed)
        if x_other_sample.shape[1] == 1:
            x_other_sample = np.sort(x_other_sample, axis=0)
        x_other_prob = np.exp(kde_x_other.score_samples(x_other_sample))
        for snum in tqdm(range(slice_num), desc=f"for x{vid}: ", leave=False):
            x_vid = np.mgrid[min(samples[:, vid]):max(samples[:, vid]):100j]
            x_all = np.zeros((len(x_vid), var_num))
            x_all[:, vid] = x_vid
            for k in range(var_num - 1):
                cur_id = k + (k >= vid)
                x_all[:, cur_id] = x_other_sample[snum][k]
            prob = np.exp(kde_all.score_samples(x_all)) # We use kde_all to score the slice of joint distribution corresponding to (x1, x2, x3)
            normalize_coe = len(x_all) / np.sum(prob) / (np.max(x_vid) - np.min(x_vid)) # This is the normalizing coefficient (area under the curve of the slice)
            prob *= normalize_coe
            file_name = f"./data/processed_data/{save_prefix}_conditional_data_{vid}_slice_{snum}.csv"
            data = np.hstack([x_vid.reshape(-1, 1), prob.reshape(-1, 1)])
            np.savetxt(file_name, data, delimiter=",")
        file_name = f"./data/processed_data/{save_prefix}_conditional_slices_{vid}.csv"
        data = np.hstack([x_other_sample.reshape(-1, 1), x_other_prob.reshape(-1, 1)])
        np.savetxt(file_name, data, delimiter=",")


def main():
    samples = read_file(DataConfig.data_file_path, DataConfig.columns)
    num_var = len(DataConfig.columns)
    generate_marginals(samples, num_var, save_prefix = DataConfig.processed_data_prefix, b_adj=DataConfig.b_adj*1.0)
    generate_conditionals(samples, save_prefix = DataConfig.processed_data_prefix, K=DataConfig.K, slice_num=DataConfig.slice_num, b_adj=DataConfig.b_adj*1.3)
    print("done!")

if __name__ == "__main__":
    main()
