import hashlib
import os
import urllib.request as request
import zipfile
from RegressionProject.entity import DataIngestionConfig
from RegressionProject.logging import logger
from datetime import datetime

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def calculate_file_hash(self, file_path):
        """Calculate the SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def download_file(self):
        print(self.config.source_URL)

        if os.path.exists(self.config.local_data_file):
            logger.info(f"File already exists of size: {os.path.getsize(self.config.local_data_file)}")
        else:
            # Download the file
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded with the following info:\n{headers}")
        # Validate file hash after download
        if self.validate_file_hash():
            print("File integrity verified.")
            download_date = datetime.now().isoformat()
            self.write_verification_status(True, download_date)
        else:
            print("File integrity verification failed.")
            self.write_verification_status(False, None)
            # Handle error or raise an exception

    def write_verification_status(self, status, download_date):
        """Write verification status and download date to a text file."""
        with open(self.config.status_file, 'w') as f:
            f.write(f"Verification Status: {'True' if status else 'False'}\n")
            if download_date:
                f.write(f"Download Date: {download_date}\n")
        print(f"Verification status written to {self.config.status_file}")

    def validate_file_hash(self):
        """Validate the hash of the downloaded file."""
        expected_hash = self.config.expected_hash  # Replace with your actual expected hash
        downloaded_hash = self.calculate_file_hash(self.config.local_data_file)
        return downloaded_hash == expected_hash

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
