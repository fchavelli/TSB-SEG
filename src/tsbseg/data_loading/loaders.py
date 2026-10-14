import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from tsseg.data.datasets import load_mocap

# Loader parameters of every benchmark series, in the order of the benchmark
# runs (``dataset_trial_index`` = ``series``), read from the directory listing
# of the cluster that ran the benchmark and checked against the runs.
SERIES_FILE = Path(__file__).with_name("series.csv")


def _benchmark_order(dataset_name: str, params_list: list[dict]) -> list[dict]:
    """``params_list`` in the order of the benchmark runs.

    The directory listings below follow the file system's order, which differs
    between machines; ``SERIES_FILE`` fixes it to the order in which the
    benchmark ran, so that a ``dataset_trial_index`` names the same series
    everywhere. A listing that does not hold exactly the benchmark series is
    returned unchanged.
    """
    if not SERIES_FILE.is_file():
        return params_list
    series = pd.read_csv(SERIES_FILE)
    series = series[series["dataset"] == dataset_name].sort_values("series")
    if series.empty:
        return params_list
    order = [json.loads(s) for s in series["source"]]

    def key(p):
        return json.dumps(p, sort_keys=True)

    if sorted(map(key, order)) != sorted(map(key, params_list)):
        warnings.warn(
            f"{dataset_name}: the files found differ from the benchmark series "
            f"({SERIES_FILE.name}); keeping the directory order",
            stacklevel=2,
        )
        return params_list
    return order


def _load_parquet(path: Path, **params):
    """Series of a converted dataset: all of them (lists), or the one ``params`` selects."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    label_dtype = np.dtype(table.schema.metadata[b"label_dtype"].decode())
    rows = range(table.num_rows)
    if params:
        wanted = json.dumps(params, sort_keys=True)
        rows = [i for i, s in enumerate(table.column("source").to_pylist()) if s == wanted]
        if len(rows) != 1:
            raise ValueError(f"{path.name}: {len(rows)} series match {params}")
    width = table.column("n_channels").to_pylist()
    X, y = [], []
    for i in rows:
        X.append(
            np.column_stack(
                [
                    table.column(f"channel_{k}")[i].values.to_numpy(zero_copy_only=False)
                    for k in range(width[i])
                ]
            ).astype(np.float64, copy=False)
        )
        labels = table.column("labels")[i].values.to_numpy(zero_copy_only=False)
        y.append(labels.astype(label_dtype, copy=False))
    if params:
        return X[0], y[0]
    return X, y


def _get_all_params(dataset_name: str, data_root: Path) -> list[dict]:
    """
    Generates a list of all valid parameter combinations for a given dataset.

    This function scans the dataset's directory structure to find all
    available time series and their corresponding parameters.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    data_root : Path
        The path to the root data directory.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary contains the parameters
        to load a single time series from the dataset.
    """
    params_list = []
    if dataset_name == "pamap2":
        protocol_path = data_root / "pamap2" / "Protocol"
        if not protocol_path.exists():
            return []
        for f in protocol_path.glob("subject10*.dat"):
            match = re.search(r"subject10(\d+)\.dat", f.name)
            if match:
                params_list.append({"subject_number": int(match.group(1))})
    elif dataset_name == "usc-had":
        base_path = data_root / "usc-had"
        if not base_path.exists():
            return []
        subject_dirs = [d for d in base_path.glob("Subject*") if d.is_dir()]
        for s_dir in subject_dirs:
            s_num_match = re.search(r"Subject(\d+)", s_dir.name)
            if not s_num_match:
                continue
            s_num = int(s_num_match.group(1))

            # Find all unique target numbers for this subject
            target_numbers = set()
            for f in s_dir.glob("a*t*.mat"):
                t_num_match = re.search(r"a\d+t(\d+)\.mat", f.name)
                if t_num_match:
                    target_numbers.add(int(t_num_match.group(1)))

            for t_num in sorted(list(target_numbers)):
                # For USC-HAD, each trial is a concatenation of all activities
                # So we just need subject and target.
                params_list.append({"subject_number": s_num, "target_number": t_num})

    elif dataset_name == "actrectut":
        base_path = data_root / "actrectut"
        if not base_path.exists():
            return []
        for d in base_path.glob("subject*_walk"):
            match = re.search(r"subject(\d+)_walk", d.name)
            if match:
                params_list.append({"subject_number": int(match.group(1))})

    elif dataset_name == "has":
        # Point to the preprocessed NPZ file.
        data_path = data_root / "has/has_data.npz"
        if not data_path.exists():
            # If the file doesn't exist, return an empty list.
            # The loader will then raise a helpful error.
            return []
        # The HAS dataset contains 250 time series with ts_id from 0 to 249.
        for i in range(250):
            params_list.append({"ts_id": i})

    elif dataset_name == "pump":
        base_path = data_root / "pump"
        if not base_path.exists():
            return []
        # Iterate over subdirectories like pump_v35, pump_v36...
        for version_dir in base_path.glob("pump_v*"):
            if version_dir.is_dir():
                version_name = version_dir.name
                for f in version_dir.glob("*.csv"):
                    params_list.append({"version": version_name, "filename": f.name})

    # For tssb, utsa, skab, mocap, the logic is already handled in their loaders
    # or they don't fit this "all params" model well without more info.
    return _benchmark_order(dataset_name, params_list)


def _load_pamap2(data_root: Path, subject_number: int):
    """Loads a trial from the PAMAP2 dataset."""
    ts_path = data_root / "pamap2" / "Protocol" / f"subject10{subject_number}.dat"
    df = pd.read_csv(ts_path, sep=" ", header=None)
    data = df.to_numpy()

    groundtruth = np.array(data[:, 1], dtype=int)
    # Columns: hand, chest, ankle accelerometer data
    hand_acc = data[:, 4:7]
    chest_acc = data[:, 21:24]
    ankle_acc = data[:, 38:41]
    X = np.hstack([hand_acc, chest_acc, ankle_acc])

    return X, groundtruth


def _load_usc_had(data_root: Path, subject_number: int, target_number: int):
    """
    Loads a single trial from the USC-HAD dataset, which consists of all
    activities concatenated for a given subject and target.
    """
    prefix = data_root / "usc-had" / f"Subject{subject_number}"

    data_list = []
    label_list = []

    # Activities are numbered 1 to 12
    for activity_number in range(1, 13):
        fname = f"a{activity_number}t{target_number}.mat"
        file_path = prefix / fname
        if not file_path.exists():
            # Some subjects may not have all activity/target combinations
            continue

        mat_data = scipy.io.loadmat(file_path)
        activity_data = mat_data["sensor_readings"]

        data_list.append(activity_data)
        label_list.append(np.full(len(activity_data), activity_number, dtype=int))

    if not data_list:
        raise FileNotFoundError(
            f"No data found for USC-HAD subject {subject_number}, target {target_number}"
        )

    X = np.vstack(data_list)
    groundtruth = np.concatenate(label_list)

    return X, groundtruth


def _load_actrectut(data_root: Path, subject_number: int):
    """Loads a trial from the ActRecTut dataset."""
    ts_path = data_root / "actrectut" / f"subject{subject_number}_walk" / "data.mat"
    mat_data = scipy.io.loadmat(ts_path)
    groundtruth = mat_data["labels"].flatten()
    X = mat_data["data"][:, 0:10]
    return X, groundtruth


def _load_tssb(data_root: Path, ts_name: str = None):
    """
    Loads time series from the TSSB dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    tssb_path = data_root / "tssb"
    desc_filename = tssb_path / "desc.txt"
    prop_filename = tssb_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        # If metadata files are missing, return empty lists or raise a more specific error
        # depending on whether we are trying to load all or one.
        if ts_name:
            raise FileNotFoundError(f"Metadata files not found in {tssb_path}")
        return [], []

    with open(desc_filename) as f:
        all_ts_info = [line.strip().split(",") for line in f]

    if ts_name:
        ts_info_list = [info for info in all_ts_info if info[0] == ts_name]
        if not ts_info_list:
            raise ValueError(f"Time series '{ts_name}' not found in TSSB dataset.")
    else:
        ts_info_list = all_ts_info

    all_X = []
    all_y = []
    for ts_info in ts_info_list:
        current_ts_name = ts_info[0]
        change_points = [int(cp) for cp in ts_info[2:] if cp]

        # Find labels from properties.txt
        labels = []
        with open(prop_filename) as file:
            for line in file:
                parts = line.strip().split(",")
                if parts[0] == current_ts_name:
                    label_cut = int(parts[2])
                    labels = [int(v) // (label_cut + 1) for v in parts[4:] if v]
                    break

        # Load time series data
        ts_filepath = tssb_path / f"{current_ts_name}.txt"
        X = np.loadtxt(fname=ts_filepath, dtype=np.float64)

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]

        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name:  # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y


def _load_utsa(data_root: Path, ts_name: str = None):
    """
    Loads time series from the UTSA dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    utsa_path = data_root / "utsa"
    desc_filename = utsa_path / "desc.txt"
    prop_filename = utsa_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        if ts_name:
            raise FileNotFoundError(f"Metadata files not found in {utsa_path}")
        return [], []

    with open(desc_filename) as f:
        all_desc_rows = [line.strip().split(",") for line in f]

    with open(prop_filename) as f:
        all_prop_rows = [line.strip().split(",") for line in f]

    if len(all_desc_rows) != len(all_prop_rows):
        raise ValueError(
            "Description and property files have a different number of records in UTSA dataset."
        )

    if ts_name:
        try:
            # Find the specific time series info
            desc_row = next(row for row in all_desc_rows if row[0] == ts_name)
            prop_row = next(row for row in all_prop_rows if row[0] == ts_name)
        except StopIteration:
            raise ValueError(f"Time series '{ts_name}' not found in UTSA dataset.") from None

        ts_info_list = [(desc_row, prop_row)]
    else:
        ts_info_list = zip(all_desc_rows, all_prop_rows)

    all_X = []
    all_y = []

    for desc_row, prop_row in ts_info_list:
        if desc_row[0] != prop_row[0]:
            raise ValueError(f"Mismatched records for '{desc_row[0]}' in UTSA dataset.")

        current_ts_name = desc_row[0]
        # desc_row[1] is window_size, which is not used here.
        change_points = [int(cp) for cp in desc_row[2:] if cp]
        labels = [int(v) for v in prop_row[1:] if v]

        # Load time series data
        ts_filepath_txt = utsa_path / f"{current_ts_name}.txt"
        ts_filepath_npz = utsa_path / "data.npz"

        if ts_filepath_txt.exists():
            X = np.loadtxt(fname=ts_filepath_txt, dtype=np.float64)
        elif ts_filepath_npz.exists():
            with np.load(file=ts_filepath_npz) as data:
                if current_ts_name in data:
                    X = data[current_ts_name]
                else:
                    raise FileNotFoundError(
                        f"Time series '{current_ts_name}' not found in data.npz for UTSA dataset."
                    )
        else:
            raise FileNotFoundError(f"Data file for '{current_ts_name}' not found in UTSA dataset.")

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]

        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name:  # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y


def _load_skab(data_root: Path, ts_name: str = None):
    """
    Loads time series from the SKAB dataset.
    If ts_name is provided, loads a single time series.
    If ts_name is None, loads all time series from the dataset.
    """
    skab_path = data_root / "skab"
    desc_filename = skab_path / "desc.txt"
    prop_filename = skab_path / "properties.txt"

    if not desc_filename.exists() or not prop_filename.exists():
        if ts_name:
            raise FileNotFoundError(f"Metadata files not found in {skab_path}")
        return [], []

    with open(desc_filename) as f:
        all_desc_rows = [line.strip().split(",") for line in f]

    with open(prop_filename) as f:
        all_prop_rows = [line.strip().split(",") for line in f]

    if len(all_desc_rows) != len(all_prop_rows):
        raise ValueError(
            "Description and property files have a different number of records in SKAB dataset."
        )

    if ts_name:
        try:
            # Find the specific time series info
            desc_row = next(row for row in all_desc_rows if row[0] == ts_name)
            prop_row = next(row for row in all_prop_rows if row[0] == ts_name)
        except StopIteration:
            raise ValueError(f"Time series '{ts_name}' not found in SKAB dataset.") from None

        ts_info_list = [(desc_row, prop_row)]
    else:
        ts_info_list = zip(all_desc_rows, all_prop_rows)

    all_X = []
    all_y = []

    for desc_row, prop_row in ts_info_list:
        if desc_row[0] != prop_row[0]:
            raise ValueError(f"Mismatched records for '{desc_row[0]}' in SKAB dataset.")

        current_ts_name = desc_row[0]
        change_points = [int(cp) for cp in desc_row[2:] if cp]
        labels = [int(v) for v in prop_row[1:] if v]

        # Load time series data from data.npz
        ts_filepath_npz = skab_path / "data.npz"
        if not ts_filepath_npz.exists():
            raise FileNotFoundError("data.npz not found for SKAB dataset.")

        with np.load(file=ts_filepath_npz) as data:
            if current_ts_name in data:
                X = data[current_ts_name]
            else:
                raise FileNotFoundError(
                    f"Time series '{current_ts_name}' not found in data.npz for SKAB dataset."
                )

        # Create groundtruth from change points and labels
        groundtruth = np.zeros(len(X), dtype=int)
        start_idx = 0
        for i, cp in enumerate(change_points):
            groundtruth[start_idx:cp] = labels[i]
            start_idx = cp
        groundtruth[start_idx:] = labels[-1]

        # Ensure X is 2D (n_timepoints, n_channels)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if ts_name:  # If a specific ts_name was requested, return single X, y
            return X, groundtruth

        all_X.append(X)
        all_y.append(groundtruth)

    return all_X, all_y


def _load_has(data_root: Path, ts_id: int):
    """Loads a time series from the preprocessed HAS dataset (.npz) by its ID."""
    # Correctly point to the preprocessed .npz file inside the 'has' subdirectory
    data_path = data_root / "has/has_data.npz"

    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. "
            "Please run the 'scripts/preprocess_has.py' script once to generate it."
        )

    with np.load(data_path) as data:
        x_key = f"X_{ts_id}"
        y_key = f"y_{ts_id}"

        if x_key not in data or y_key not in data:
            raise ValueError(f"Time series with ts_id {ts_id} not found in {data_path}.")

        X = data[x_key]
        groundtruth = data[y_key]

    return X, groundtruth


def _load_pump(data_root: Path, version: str, filename: str):
    """
    Loads a single trial from the Pump dataset.

    Parameters
    ----------
    data_root : Path
        Root data directory.
    version : str
        The version folder name (e.g. 'pump_v35').
    filename : str
        The CSV filename (e.g. 'Pump_A_DC35_0.csv').

    Returns
    -------
    X : np.ndarray
        Array of sensor readings (Sensor 1..9).
    y : np.ndarray
        Array of state labels.
    """
    file_path = data_root / "pump" / version / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Pump file not found: {file_path}")

    # Read CSV. Header is presumed to exist.
    df = pd.read_csv(file_path)

    # Last column is 'state', others are sensors
    # Check if 'state' is in columns, else assume last column
    if "state" in df.columns:
        y = df["state"].to_numpy(dtype=int)
        X = df.drop(columns=["state"]).to_numpy()
    else:
        # Fallback if header is missing or different
        data = df.to_numpy()
        X = data[:, :-1]
        y = data[:, -1].astype(int)

    return X, y


def load_dataset(dataset_name: str, data_root: str = "data/", return_X_y=True, **params):
    """
    Loads time series data from a specified dataset.

    This function acts as a dispatcher. If specific parameters (e.g.,
    `subject_number`, `ts_name`) are provided, it loads a single time series.
    If no parameters are given, it attempts to load all available time series
    from the dataset, returning them as lists of arrays.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., 'pamap2', 'tssb').
    data_root : str, default="data/"
        The path to the root directory containing all dataset folders.
    return_X_y : bool, default=True
        If True, returns (X, y) tuple.
    **params : dict
        Keyword arguments specific to the dataset loader. If empty, the function
        will try to load all time series from the dataset.
        - 'pamap2': `subject_number` (int)
        - 'usc-had': `subject_number` (int), `target_number` (int)
        - 'actrectut': `subject_number` (int)
        - 'tssb', 'utsa', 'skab': `ts_name` (str, optional)
        - 'has': `ts_id` (int)
        - 'mocap': `trial` (int or str, optional)

    Returns
    -------
    X : np.ndarray or list of np.ndarray
    y : np.ndarray or list of np.ndarray
        Ground truth labels. A list is returned if all series are loaded.
    """
    data_root_path = Path(data_root)
    # The converted dataset (one Parquet file per dataset, see
    # benchmark/datasets/to_parquet.py) takes precedence over the source layout.
    parquet = data_root_path / f"{dataset_name}.parquet"
    if parquet.is_file():
        return _load_parquet(parquet, **params)

    # If no specific parameters are passed, try to load all time series
    if not params:
        # Datasets whose loaders read every series at once
        if dataset_name in ["tssb", "utsa", "skab"]:
            if dataset_name == "tssb":
                X, y = _load_tssb(data_root_path)
            elif dataset_name == "utsa":
                X, y = _load_utsa(data_root_path)
            elif dataset_name == "skab":
                X, y = _load_skab(data_root_path)
        elif dataset_name == "mocap":
            # load_mocap reads one trial at a time: iterate until the index is invalid.
            all_X, all_y = [], []
            trial_idx = 0
            while True:
                try:
                    # Recursively call this function with a specific trial index
                    single_X, single_y = load_dataset(
                        dataset_name, data_root, return_X_y=True, trial=trial_idx
                    )
                    all_X.append(single_X)
                    all_y.append(single_y)
                    trial_idx += 1
                except ValueError as e:
                    # An invalid trial index ends the iteration.
                    if "Invalid trial index" in str(e):
                        break
                    else:
                        raise e  # Re-raise other ValueErrors
            if not all_X:
                raise ValueError("Could not load any trials for mocap dataset.")
            X, y = all_X, all_y
        else:
            # Use the helper to find all parameter combinations
            all_params = _get_all_params(dataset_name, data_root_path)
            if not all_params:
                raise ValueError(
                    f"Could not find any time series for dataset '{dataset_name}'. "
                    "Please provide specific parameters to load a single series."
                )

            all_X, all_y = [], []
            for p in all_params:
                # Recursively call this function with specific params for one series
                single_X, single_y = load_dataset(dataset_name, data_root, return_X_y=True, **p)
                all_X.append(single_X)
                all_y.append(single_y)
            X, y = all_X, all_y
    else:
        # Load a single time series using provided parameters
        if dataset_name == "pamap2":
            X, y = _load_pamap2(data_root_path, **params)
        elif dataset_name == "usc-had":
            X, y = _load_usc_had(data_root_path, **params)
        elif dataset_name == "actrectut":
            X, y = _load_actrectut(data_root_path, **params)
        elif dataset_name == "tssb":
            X, y = _load_tssb(data_root_path, **params)
        elif dataset_name == "utsa":
            X, y = _load_utsa(data_root_path, **params)
        elif dataset_name == "skab":
            X, y = _load_skab(data_root_path, **params)
        elif dataset_name == "has":
            X, y = _load_has(data_root_path, **params)
        elif dataset_name == "pump":
            X, y = _load_pump(data_root_path, **params)
        elif dataset_name == "mocap":
            # The load_mocap function from tsseg does not use data_root
            X, y = load_mocap(**params, return_X_y=True)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported by this loader.")

    if return_X_y:
        return X, y
    else:
        if isinstance(X, list):
            raise ValueError("Cannot return as DataFrame when multiple series are loaded.")
        return pd.DataFrame(X), pd.Series(y)
