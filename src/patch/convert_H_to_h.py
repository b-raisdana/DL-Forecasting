import hashlib
import os
import re
import zipfile

from Config import config  # Make sure to import your config module
from data_processing.fragmented_data import data_path
from helper.helper import log_d


def convert_hour_timeframes_case():
    """
    We have to check all files in file_path = data_path directory names ad multi_timeframe_*.zip.
    These zip files include a .csv file.
    Replace all the whole word matches for timeframes listed in confi.timeframes (which include h like 1h or 4h)
    but in cap[ital case with lowercase h. as an example covert 1H to 1h and convert 4H to 4h.
    Do not convert if the match is not a whole word.
    """

    def process_zip_file(zip_file_path, timeframe_with_h):
        file_to_write = ""
        modified_content = ""
        with zipfile.ZipFile(zip_file_path, 'a') as zip_ref:
            for file_info in zip_ref.infolist():
                # Read the content of the CSV file
                with zip_ref.open(file_info.filename, 'r') as csv_file:
                    original_content = csv_file.read().decode('utf-8')
                    modified_content = original_content
                    original_hash = hashlib.sha256(original_content.encode('utf-8')).hexdigest()
                        # Replace timeframes in the content
                    for timeframe in timeframe_with_h:
                        # Use regular expression to match whole words
                        pattern = r'\b{}\b'.format(re.escape(timeframe.upper()))
                        replacement = timeframe.lower()
                        modified_content = re.sub(pattern, replacement, modified_content)
                    modified_hash = hashlib.sha256(modified_content.encode('utf-8')).hexdigest()
                if original_hash != modified_hash:
                    file_to_write = file_info.filename
                    log_d(f'{zip_file_path} altered.')
                    # Open the zip file in write mode to overwrite the existing file
                    # Write the modified content back to the temporary file
        if file_to_write != "":
            with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
                # Add a file to the zip file with the modified content
                zip_file.writestr(file_to_write, modified_content)
    def process_all_zip_files(directory_path):
        timeframe_with_h = [timeframe for timeframe in config.timeframes if 'h' in timeframe.lower()]
        for filename in os.listdir(directory_path):
            if filename.startswith('multi_timeframe_') and filename.endswith('.zip'):
                zip_file_path = os.path.join(directory_path, filename)
                process_zip_file(zip_file_path, timeframe_with_h)

    file_path = data_path()
    process_all_zip_files(file_path)


convert_hour_timeframes_case()
