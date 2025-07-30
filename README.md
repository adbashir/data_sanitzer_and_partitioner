# Triton Sensor Data Cleansing & Partitioning Pipeline
## Adnan Bashir - UNM & State of NM
### 30th July 2025

This repository contains a Python script to preprocess, clean, and partition IoT sensor data (Heartbeats and Alerts) from Triton ULTRA and Triton PRO devices. The script prepares data for scalable analytics, AWS S3 upload, and AWS Glue/Athena consumption.

## Features

- Cleans and type-corrects raw CSVs (custom headers & types).
- Aggregates data into 10-minute buckets (mode for discrete/integer columns, mean for floats).
- Handles missing values (interpolation and mode-filling).
- Partitions output by `device_id` and `date` into AWS Glue/Athena–friendly folders and files.
- Outputs progress with a CLI progress bar.

## Directory Structure

- Input files:  
  - `../data/raw_alerts.csv`
  - `../data/raw_heartbeats.csv`
- Output directory (default):  
  - `data/partitioned_data/alerts/device_id=.../date=YYYY-MM-DD/alerts.csv`
  - `data/partitioned_data/heartbeats/device_id=.../date=YYYY-MM-DD/heartbeats.csv`

## Requirements

See [requirements.txt](./requirements.txt) for dependencies.

- Python 3.8+
- Key packages: `pandas`, `numpy`, `scipy`, `tqdm`

## Installation

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
