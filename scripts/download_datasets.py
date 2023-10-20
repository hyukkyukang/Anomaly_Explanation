import argparse
import os
import zipfile

import gdown

# Link
datasets = {
    "SMD": "https://drive.google.com/file/d/187cjlXCedf4v3-Xm-fK7iTQZ6k6BFaDK/view?usp=sharing",
    "SMAP": "https://drive.google.com/file/d/1DRj6A4wFGx7SNEalGkEPzRi8Td8RdEK1/view?usp=sharing",
    "PSM": "https://drive.google.com/file/d/1kohMqejb7f787XtpM4b5HR7G22nH-rEF/view?usp=sharing",
    "MSL": "https://drive.google.com/file/d/1BGeu0yiV4T_nsI1G2ayuGfLjIKw9ArZ_/view?usp=sharing",
}


def main(output_dir: str) -> None:
    for name, link in datasets.items():
        download_path = os.path.join(output_dir, f"{name}.zip")
        print(f"Downloading {name} dataset...")
        gdown.download(link, download_path, quiet=False, fuzzy=True)
        unzip_path = download_path[:-4]
        print("Extracting...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        print("Removing zip file...")
        os.remove(download_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/root/Anomaly_Explanation/dataset/",
        help="path to save the dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    main(output_dir=output_dir)
