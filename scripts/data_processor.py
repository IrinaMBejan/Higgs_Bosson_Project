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
    """
    Split data on the test and train data sets.
    Args:
        x: the input dataset
        y: the labels for the input dataset
        ratio: float number between 0 to 1 to specify the percentage of training data
        seed: seed number to get same permutation of the data
    """
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
    """
    Preprocess input data
    Args:
        tx: The input dataset
        y: Labels of the dataset
        use_dropping: If true, drop columns with missing values
        remove_outliers: If true, any values that lie 4.5IQR below Q1 or above Q3 are removed.
        usePCA: If true, principal component analysis is done and highly correlated features are removed.
        poly_rank: The degree of the polynomial expansion. If None, the expansion is not applied.
        use_log: If true, expands the dataset with the logarithmic scale of all features.
        log_mean: The mean value of the logarithmic scale features. If not specified, it is computed for given values.
        log_std: The standard deviation of features. If not specified, it is computed for given values.
        mean: The mean of given values without the log scale. If not specified, it is computed for the given values.
        std: The standard dev of given values without std scale. If not specified, it is computed for the given values.
    """

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

    if usePCA:  # Do Principal Component Analysis
        eig_val, eig_vec, j = PCA(tx, 0.97)
        tx = tx.dot(eig_vec)
        print('Columns left:', j)

    if poly_rank:  # Build polynomial basis function
        tx = build_poly(tx, poly_rank)

    return tx, y, mean, std, log_mean, log_std


def get_with_jet(dataset, outputs, jet_num):
    """
    Computes the subdataset according to a given jet.
    Args:
        dataset: The dataset to be split. The column 22 must contain the jet number.
        outputs: The labels of the dataset.
        jet_num: The jet number (must be 0,1,2 or 3).

    Returns:
        A subset of the initial dataset
    """
    dataset[:, 22] = np.where(dataset[:, 22] > 3, 3, dataset[:, 22])

    rows = dataset[:, 22] == jet_num
    if outputs.size != 0:
        return dataset[rows, :], outputs[rows], rows
    else:
        return dataset[rows, :], np.array([]), rows


def columns_to_drop(jet_num):
    """
    Columns to drop in each of the jet sets.
    Args:
        jet_num: Integer presenting the jet number.
    Returns:
        A list of columns indices to be dropped from the initial dataset.
    """

    columns = []

    if jet_num == 0:
        columns = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    elif jet_num == 1:
        columns = [4, 5, 6, 12, 22, 26, 27, 28]
    else:
        columns = [22]

    return columns


def split_input_data(dataset, output=np.array([])):
    """Split input data into jets."""
    num_jets = 4
    datasets = {}
    outputs = {}
    rows = {}
    for jet in range(num_jets):
        sub_dataset, sub_output, row = get_with_jet(dataset, output, jet)
        sub_dataset = np.delete(sub_dataset, columns_to_drop(jet), axis=1)

        datasets[jet] = sub_dataset
        outputs[jet] = sub_output
        rows[jet] = row

    return datasets, outputs, rows


def get_outlier_mask(feature_column):
    """
    Computes a mask to show what samples in the given feature are outliers.
    Args:
        feature_column: A 1-D numpy array representing a feature column from initial dataset
    Returns:
        A mask with the same shape as the feature column
    """
    Q1 = np.nanquantile(feature_column, .25)
    Q3 = np.nanquantile(feature_column, .75)
    IQR = Q3 - Q1

    def is_outlier(value):
        # An outlier is considered to be part of the the lower and upper 5% intervals of the distribution
        return (value < (Q1 - 4.5 * IQR)) | (value > (Q3 + 4.5 * IQR))

    is_outlier_map = np.vectorize(is_outlier)
    return is_outlier_map(feature_column)


def remove_outlier_points(data, labels):
    """
    Removes the outliers from data and labels accordingly based on the IQR method.
    """
    feature_columns_masks = np.stack([get_outlier_mask(data[:, i]) for i in range(data.shape[1])])
    datapoints_masks = feature_columns_masks.T
    outliers = np.array([np.any(point) for point in datapoints_masks])
    return data[~outliers], labels[~outliers]


def PCA(tx, threshold):
    """
    Performs the Principal Component Analysis method, computing the covariance matrix.
    Args:
        tx: the dataset
        threshold: The percentage under which the components should be captured.
    Returns:
        The eigen values, eigen vectors and the number of features above the threshold.
    """
    cov_matrix = np.cov(tx.T)  # computing covariance matrix, this represents the correlation between two variables
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)  # eigenvalues and eignvectors

    # sort eigenvalues in decreasing order
    idx = eig_values.argsort()[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    eig_values = eig_values / sum(eig_values)

    # get only the components that capture certain percent of overall variance
    sum_ = 0
    k = 0
    while (sum_ < threshold):
        sum_ = sum_ + eig_values[k]
        k += 1

    eig_vectors = eig_vectors[:, :k - 1]

    return eig_values, eig_vectors, k - 1


def cap_outliers_fn(x):
    """Caps the outliers of a subset of features based on observation on data points plots"""
    capping_values = {
        0: 700,   # DER_mass_MMC
        1: 250,   # DER_mass_transverse_met_lep
        2: 550,   # DER_mas_vis
        3:500,    # DER_pt_h
        8: 300,   # DER_pt_tot
        9: 900,   # DER_sum_pt
        10: 9.5,  # DER_pt_ratio_lep_tau
        13: 280,  # PRI_tau_pt
        16: 230,  # PRI_lep_pt
        19: 375,  # PRI_met
        21: 1000,  # PRI_met_sumet
        23: 475,  # PRI_jet_leading_pt
        26: 240,  # PRI_jet_subleading_pt
        29: 775   # PRI_jet_all_pt

    }
    
    for key, value in capping_values.items():
        x[x[:, key] > value] = value

    return x
