import os
import json

# Configuration
CONFIG_FILE = "xgb_cpu_main_config.json"
DATASET_FOLDER = "dataset"
EXPECTED_DATASETS = ["mlsr", "mortgage1Q", "plasticc", "santander"]

def load_config():
    """Load the benchmark configuration file."""
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: Configuration file '{CONFIG_FILE}' not found. Verify its location.")
        return None

    with open(CONFIG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: Failed to parse '{CONFIG_FILE}'. Ensure it contains valid JSON.")
            return None

def check_datasets():
    """Check if required datasets exist in the dataset folder."""
    missing_datasets = []
    for dataset in EXPECTED_DATASETS:
        dataset_path = os.path.join(DATASET_FOLDER, dataset)
        if not os.path.exists(dataset_path):
            print(f"⚠️ WARNING: Dataset '{dataset}' is missing in '{DATASET_FOLDER}'.")
            missing_datasets.append(dataset)

    if missing_datasets:
        print("\n🔹 Suggested Actions:")
        print("- Ensure dataset names are correct in the 'dataset/' folder.")
        print("- Download the missing datasets if necessary.")
        print("- If dataset names differ, update 'xgb_cpu_main_config.json'.\n")

    return missing_datasets

def update_config(missing_datasets):
    """Fix dataset names in the configuration file if necessary."""
    config = load_config()
    if not config:
        return

    updated = False
    for dataset in missing_datasets:
        if dataset in config.get("datasets", {}):
            print(f"🛠️ Fixing dataset path for '{dataset}' in {CONFIG_FILE}...")
            config["datasets"][dataset] = os.path.join(DATASET_FOLDER, f"{dataset}.csv")  # Adjust extension if necessary
            updated = True

    if updated:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        print(f"✅ {CONFIG_FILE} has been updated with corrected dataset paths.")

if __name__ == "__main__":
    print("🔍 Checking dataset availability...\n")
    missing = check_datasets()

    if missing:
        update_config(missing)
    else:
        print("✅ All datasets are present. You can proceed with benchmarking.")
