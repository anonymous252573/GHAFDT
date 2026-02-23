import os
import shutil
from collections import defaultdict

def collect_and_split_subjects(base_dir, groups=("AD", "CN"), val_ratio=0.2):

    subject_dict = {}
    all_timepoints = set()

    for group in groups:
        group_dir = os.path.join(base_dir, group)

        if not os.path.isdir(group_dir):
            raise FileNotFoundError(f"Missing folder: {group_dir}")

        for fname in os.listdir(group_dir):
            if not fname.endswith(".txt"):
                continue

            name = fname.replace(".txt", "")
            parts = name.split("_")

            subject_id = "_".join(parts[1:-1])
            timepoint = parts[-1]

            all_timepoints.add(timepoint)

            if subject_id not in subject_dict:
                subject_dict[subject_id] = {
                    "group": group,
                    "files": {}
                }

            subject_dict[subject_id]["files"][timepoint] = os.path.join(group_dir, fname)

    # Split subjects by group
    group_subjects = defaultdict(list)
    for sid, info in subject_dict.items():
        group_subjects[info["group"]].append((sid, len(info["files"])))

    # Sort subjects by number of timepoints (descending)
    for group in groups:
        group_subjects[group].sort(key=lambda x: x[1], reverse=True)

    # Assign to validation/test
    split_dict = {}
    for group in groups:
        subjects = [sid for sid, _ in group_subjects[group]]
        n_val = max(1, int(len(subjects) * val_ratio))  # at least 1

        val_subjects = subjects[:n_val]
        test_subjects = subjects[n_val:]

        split_dict[group] = {
            "val": val_subjects,
            "test": test_subjects
        }

    # Create folders and copy files
    for group in groups:
        for split in ["val", "test"]:
            out_dir = os.path.join(base_dir, f"{group}_{split}")
            os.makedirs(out_dir, exist_ok=True)

            for sid in split_dict[group][split]:
                for tp, fpath in subject_dict[sid]["files"].items():
                    shutil.copy(fpath, out_dir)

    print("Done! Folders created and files copied:")
    for group in groups:
        for split in ["val", "test"]:
            print(f"{group}_{split}: {len(split_dict[group][split])} subjects")

    return split_dict


# Main execution
if __name__ == "__main__":

    base_dir = r"C:\....\ADNI_split"
    collect_and_split_subjects(base_dir, groups=("AD", "CN"), val_ratio=0.2)