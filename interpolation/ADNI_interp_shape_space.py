import os
import shutil
import numpy as np
from geomstats.geometry.pre_shape import PreShapeSpace
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.backend as gs
from collections import defaultdict
import pickle

m, k = 732, 3
shape_space = PreShapeSpace(m, k)

def frobenius_norm(matrix):
    return np.linalg.norm(matrix, ord='fro')

def load_subject_shapes(folder_path):
    subjects = {}
    before_matrix = None
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        
        name = fname.replace(".txt", "")
        parts = name.split("_")
        subject_id = "_".join(parts[1:-1]) 
        timepoint = parts[-1]
        
        full_path = os.path.join(folder_path, fname)
        before_vectors = []
        after_vectors = []
        with open(full_path, 'r') as f:
            for line in f:
                before, after = line.split('->')
                if before_matrix is None:
                    before_vectors.append(
                        np.array([float(x) for x in before.strip().split()]))
                after_vectors.append(
                    np.array([float(x) for x in after.strip().split()]) )

        if before_matrix is None:
            before_matrix = np.array(before_vectors) 

        shape = np.array(after_vectors) / frobenius_norm(after_vectors)
        subjects.setdefault(subject_id, {})[timepoint] = shape
    return subjects, before_matrix


def process_and_interpolate_subject(subject_shapes, shape_space):
    required = {'bl'}
    if not required.issubset(subject_shapes.keys()):
        return {}

    sorted_keys = sorted(subject_shapes.keys(), key=lambda x: 1 if x=='bl' else int(x[1:]))
    subject_shapes = {k: subject_shapes[k] for k in sorted_keys}

    times = ['bl', 'm03', 'm06', 'm12']
    time_map = {'bl': 1., 'm03':3., 'm06':6., 'm12':12.}

    # Check if all main time points exist
    if all(t in subject_shapes for t in times):
        return {t: subject_shapes[t] for t in times}

    available_main_times = [t for t in times if t in subject_shapes]
    if available_main_times == ['bl'] or available_main_times == ['bl', 'm03']:
        return {t: subject_shapes[t] for t in available_main_times}

    # Interpolation
    existing_times = [t for t in times if t in subject_shapes]
    shapes = [subject_shapes[t] for t in existing_times]
    t_obs = np.array([time_map[t] for t in existing_times])

    # Karcher mean
    shape_array = gs.array(shapes)
    frechet_mean = FrechetMean(space=shape_space)
    mean_shape = frechet_mean.fit(shape_array).estimate_

    # Log maps
    V = gs.stack([shape_space.metric.log(s, mean_shape) for s in shapes])
    n_obs = V.shape[0]

    # Normalize time
    t_mean, t_std = t_obs.mean(), t_obs.std()
    t_norm = (t_obs - t_mean) / t_std

    if n_obs == 2:
        T = gs.stack([gs.ones_like(t_norm), t_norm], axis=1)
    else:
        T = gs.stack([gs.ones_like(t_norm), t_norm, t_norm**2], axis=1)

    V_flat = V.reshape(n_obs, -1)
    coeffs, _, _, _ = np.linalg.lstsq(T, V_flat, rcond=None)
    coeffs = coeffs.reshape(coeffs.shape[0], *V.shape[1:])

    if n_obs == 2:
        v0, v1 = coeffs
        def predict(month):
            t = (month - t_mean) / t_std
            Vt = v0 + v1 * t
            return shape_space.metric.exp(Vt, mean_shape)
    else:
        v0, v1, v2 = coeffs
        def predict(month):
            t = (month - t_mean) / t_std
            Vt = v0 + v1 * t + v2 * t**2
            return shape_space.metric.exp(Vt, mean_shape)

    # Fill missing time points with interpolation
    all_shapes = {t: subject_shapes[t] for t in existing_times}
    missing_tps = [
                    t for t in times
                    if t not in subject_shapes
                    and time_map[t] > time_map[existing_times[0]]
                    and time_map[t] < time_map[existing_times[-1]]
                    ]
    for miss in missing_tps:
        all_shapes[miss] = predict(time_map[miss])

    return all_shapes


def save_shape_txt(before_matrix, after_matrix, save_path):
    with open(save_path, 'w') as f:
        # print(before_matrix.shape, after_matrix.shape)
        for b, a in zip(before_matrix, after_matrix):
            b_str = " ".join(map(str, b))
            a_str = " ".join(map(str, a))
            f.write(f"{b_str} -> {a_str}\n")

condis = ['AD', 'CN']
for condi in condis:

    output_root = r"C:\...\After_matching_DG"
    splits_root = r"C:\...\ADNI\After_matching"

    folder_path = os.path.join(splits_root, condi)
    save_folder = os.path.join(output_root, condi)
    os.makedirs(save_folder, exist_ok=True)

    subjects, before_matrix = load_subject_shapes(folder_path)

    for subject_id, shapes in subjects.items():
        
        # Interpolate if condition met and save original and interpolations
        interp_shapes = process_and_interpolate_subject(shapes, shape_space)

        for tp, shape in interp_shapes.items():
            fname = f"matched_{subject_id}_{tp}.txt"
            save_shape_txt(
                before_matrix,
                shape,
                os.path.join(save_folder, fname)
            )
