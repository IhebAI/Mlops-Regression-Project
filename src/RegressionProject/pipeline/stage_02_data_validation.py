import sys
from pathlib import Path

from RegressionProject.config.configuration import ConfigurationManager
from RegressionProject.components.data_validation import DataValidation
from RegressionProject.logging import logger

STAGE_NAME = "Data Validation stage"


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            # Read the status from the file
            status_file_path = Path("artifacts/data_ingestion/status.txt")
            if status_file_path.exists():
                with open(status_file_path, "r") as f:
                    lines = f.readlines()
                    # Extract the verification status
                    status_line = next((line for line in lines if "Verification Status" in line), None)
                    if status_line:
                        status = status_line.split(":")[-1].strip()
                        print("status", status)

                        # Check the status and proceed accordingly
                        if status == "True":
                            config = ConfigurationManager()
                            data_validation_config = config.get_data_validation_config()
                            data_validation = DataValidation(config=data_validation_config)
                            data_validation.validate_all_columns()
                        else:
                            sys.exit("Your raw data has been modified")
                    else:
                        sys.exit("Verification Status not found in status file")
            else:
                sys.exit("Status file does not exist")

        except Exception as e:
            print(e)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
