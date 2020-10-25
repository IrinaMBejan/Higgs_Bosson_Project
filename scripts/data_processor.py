import numpy as np


def standardize(x, mean=None, std=None):
    """Standardize the data"""
    if mean is None:
        mean = np.mean(x, axis=0)

    centered_data = x - mean

    if std is None:
        std = np.std(centered_data, axis=0)

    std_data = centered_data / std
    return std_data, mean, std


def scaling(x, min_x=None, max_x=None):
    """Scale the data from 0 to 1"""
    if min_x is None:
        min_x = np.min(x, axis=0)

    if max_x is None:
        max_x = np.max(x, axis=0)

    scaled_data = (x - min_x) / (max_x - min_x)
    return scaled_data


def split_data(x, y, ratio, seed=1):
    """Split data on the test and train data sets"""
    # set seed
    np.random.seed(seed)
    row_num = len(y)

    # permute inputs
    perm = np.random.permutation(row_num)
    index_split = int(ratio * row_num)

    x = x[perm]
    y = y[perm]

    # split
    x_train, x_test, _ = np.split(x, [index_split, row_num])
    y_train, y_test, _ = np.split(y, [index_split, row_num])

    return x_train, x_test, y_train, y_test


def replace_with_mean(feature_column):
    """"Replace NaN values with the mean of the feature column"""
    mean = np.nanmean(feature_column.astype('float64'))
    feature_column = np.where(np.isnan(feature_column),
                              mean,
                              feature_column)
    return feature_column


def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))

    for i in range(degree):
        poly = np.c_[poly, np.power(x, i + 1)]

    return poly


def preprocess_inputs(tx, y, use_dropping=False, remove_outliers=False, usePCA=False, poly_rank=None, use_log=True,
                      log_mean=None, log_std=None, mean=None, std=None):
    """Preprocess input data"""

    tx = np.where(tx == -999, np.nan, tx)  # replace -999 values with the mean

    if remove_outliers:  # remove outliers
        tx, y = remove_outlier_points(tx, y)

    if use_dropping:  # drop columns
        columns_to_drop = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
        tx = np.delete(tx, columns_to_drop, axis=1)
    else:
        for i in range(tx.shape[1]):
            tx[:, i] = replace_with_mean(tx[:, i])  # replace NaN values with the mean
        # tx = np.nan_to_num(tx)

    if use_log:
        # Standardize & scale the data
        eps = 1e-320
        log_tx = np.abs(tx)
        log_tx[log_tx < eps] = 1
        log_tx = np.log(log_tx)
        log_tx, log_mean, log_std = standardize(log_tx, mean=log_mean, std=log_std)
        log_tx = scaling(log_tx)

    tx, mean, std = standardize(tx, mean=mean, std=std)
    tx = scaling(tx)

    if use_log:
        tx = np.c_[tx, log_tx]

    if usePCA:  # Primal Component Analysis
        eig_val, eig_vec, j = PCA(tx, 0.97)
        tx = tx.dot(eig_vec)
        print('Columns left:', j)

    if poly_rank:  # Build polynomial basis function
        tx = build_poly(tx, poly_rank)

    return tx, y, mean, std, log_mean, log_std


def get_with_jet(dataset, output_all, jet_num):
    "Given jet and dataset return the rows with th egiven jet number"
    dataset[:, 22] = np.where(dataset[:, 22] > 3, 3, dataset[:, 22])

    rows = dataset[:, 22] == jet_num
    if output_all.size != 0:
        return dataset[rows, :], output_all[rows], rows
    else:
        return dataset[rows, :], np.array([]), rows


def columns_to_drop(jet_idx):
    "Columns to drop in each of the jet sets"
    to_drop = []

    if jet_idx == 0:
        to_drop = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]  # 22 and 29 contains only 0s, the others only -999
    elif jet_idx == 1:
        to_drop = [4, 5, 6, 12, 22, 26, 27, 28]
    else:
        to_drop = [22]

    return to_drop


def split_input_data(dataset_all, output_all=np.array([])):
    "Split input data into jets"
    num_jets = 4
    datasets = {}
    outputs = {}
    original_columns = {}  # for every dataset keep which are the original columns
    rows = {}
    for jet in range(num_jets):
        curr_dataset, curr_output, row = get_with_jet(dataset_all, output_all, jet)

        # drop columns depending on the jet, drop always the PRI_jet_num (column 22)
        curr_dataset = np.delete(curr_dataset, columns_to_drop(jet), axis=1)

        datasets[jet] = curr_dataset
        outputs[jet] = curr_output
        rows[jet] = row

    return datasets, outputs, rows


def get_outlier_mask(feature_column):
    "Mask for the outliers"
    Q1 = np.nanquantile(feature_column, .25)
    Q3 = np.nanquantile(feature_column, .75)
    IQR = Q3 - Q1

    def is_outlier(value):
        return (value < (Q1 - 5 * IQR)) | (value > (Q3 + 5 * IQR))

    is_outlier_map = np.vectorize(is_outlier)
    return is_outlier_map(feature_column)


def remove_outlier_points(data, labels):
    """Remove outliers"""
    feature_columns_masks = np.stack([get_outlier_mask(data[:, i]) for i in range(data.shape[1])])
    datapoints_masks = feature_columns_masks.T
    outliers = np.array([np.any(point) for point in datapoints_masks])
    return data[~outliers], labels[~outliers]


def PCA(tx, treshold):
    """ Principal Component Analysis """

    cov_matrix = np.cov(tx.T)  # computing covariance matrix, this represents the correlation between two variables
    eig_vals, eig_ves = np.linalg.eig(cov_matrix)  # eigenvalues and eignvectors

    # sort eigenvalues in decreasing order
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_ves = eig_ves[:, idx]

    eig_vals = eig_vals / sum(eig_vals)

    # get only the components that capture certain percent of overall variance
    sum_ = 0
    k = 0
    while (sum_ < treshold):
        sum_ = sum_ + eig_vals[k]
        k += 1

    eig_ves = eig_ves[:, :k - 1]

    return eig_vals, eig_ves, k - 1


def cap_outliers_fn(x):
    "Capping the outliers"
    print("Capping the outliers")
    # DER_mass_MMC
    x[x[:, 0] > 700] = 700

    # DER_mass_transverse_met_lep
    x[x[:, 1] > 250] = 250

    # DER_mas_vis
    x[x[:, 2] > 550] = 550

    # DER_pt_h
    x[x[:, 3] > 500] = 500

    # DER_mass_jet_jet
    x[x[:, 5] > 2500] = 2500

    # DER_pt_tot
    x[x[:, 8] > 300] = 300

    # DER_sum_pt
    x[x[:, 9] > 900] = 900

    # DER_pt_ratio_lep_tau
    x[x[:, 10] > 9.5] = 9.5

    # PRI_tau_pt
    x[x[:, 13] > 280] = 280

    # PRI_lep_pt
    x[x[:, 16] > 230] = 230

    # PRI_met
    x[x[:, 19] > 375] = 375

    # PRI_met_sumet
    x[x[:, 21] > 1000] = 1000

    # PRI_jet_leading_pt
    x[x[:, 23] > 475] = 475

    # PRI_jet_subleading_pt
    x[x[:, 26] > 240] = 240

    # PRI_jet_all_pt
    x[x[:, 29] > 775] = 775

    return x
