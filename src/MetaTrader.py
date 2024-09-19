from __future__ import annotations

import os
import struct
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from os import path, remove
from pathlib import Path
from typing import Union, List

import psutil
from pandera import typing as pt

from PanderaDFM.OHLCV import OHLCV
from helper.data_preparation import map_symbol, extract_file_info, FileInfoSet
from helper.helper import log, LogSeverity

# def mt5_client() -> Mt5:
#     global META_TRADER_IS_INITIALIZED
#     if not META_TRADER_IS_INITIALIZED:
#         if not Mt5.initialize():
#             print("initialize() failed, error code =", Mt5.last_error())
#             quit()
#         META_TRADER_IS_INITIALIZED = True
#     return Mt5


_meta_trader_symbol_map = {
    'BTC/USDT': 'BTCUSDT',
}


def tree_to_dict(element: ET.Element) -> dict:
    output = {}
    for key, value in element.attrib:
        output[key] = value
    for child in element:
        if child.tag in output.items():
            raise Exception(f'Child name {child.tag} conflicts with attributes: {output}')
        output[child.tag] = tree_to_dict(child)
    return output


def compare_tree(left: ET.Element, right: ET.Element, path_to_root: List[str] = [], debug: bool = False) -> bool:
    compare_ignored_keys = [
        'id',
        'description',
        'digits',
        'position_time',
        'scale_fixed_min',
        'scale_fixed_max',
        'shift_size',
        'window_type',
    ]
    result = True
    if left.tag != right.tag:
        if debug: log(f"{path_to_root}->{left.tag}: Different tags ({left.tag} / {right.tag})", stack_trace=False)
        result = False
    left_only_attributes = set(left.attrib.keys()).difference(right.attrib.keys())
    if len(left_only_attributes) > 0:
        if debug: log(f"{path_to_root}->{left.tag}[{','.join(left_only_attributes)}]: "
                      f"attributes only in right_element!", stack_trace=False)
        result = False
    right_only_attributes = set(right.attrib.keys()).difference(left.attrib.keys())
    if len(right_only_attributes) > 0:
        if debug: log(f"{path_to_root}->{left.tag}[{','.join(right_only_attributes)}]: "
                      f"attributes only in right_element!", stack_trace=False)
        result = False
    for key in set(left.attrib.keys()).intersection(right.attrib.keys()):
        if key not in compare_ignored_keys and left.attrib[key] != right.attrib[key]:
            if debug: log(f"{path_to_root}->{left.tag}[{key}]: "
                          f"\{left.attrib[key]} != {right.attrib[key]}", stack_trace=False)
            result = False
    left_as_dict = {}
    for child in left:
        left_as_dict[child.tag] = child
    right_as_dict = {}
    for child in right:
        right_as_dict[child.tag] = child
    left_only_children_tags = set(left_as_dict.keys()).difference(right_as_dict.keys())
    if len(left_only_children_tags) > 0:
        if debug: log(f"{path_to_root}->{left.tag}[{','.join(left_only_children_tags)}]: "
                      f"children only in left_element!", stack_trace=False)
        result = False
    right_only_children_tags = set(right_as_dict.keys()).difference(left_as_dict.keys())
    if len(right_only_children_tags) > 0:
        if debug: log(f"{path_to_root}->{left.tag}[{','.join(right_only_children_tags)}]: "
                      f"children only in right_element!", stack_trace=False)
        result = False
    for key in set(left_as_dict.keys()).intersection(set(right_as_dict.keys())):
        result = result and compare_tree(left_as_dict[key], right_as_dict[key], path_to_root + [left.tag])
    return result


def check_file_sizes(directory='./MetaTrader5/Bases/Custom/history/CustomBTCUSDT', max_size_mb=100,
                     raise_exception: bool = True):
    """
    Check if all files in the given directory are less than the specified size.

    Args:
    directory (str): The directory to check.
    max_size_mb (int): The maximum file size in MB.

    Returns:
    bool: True if all files are less than max_size_mb, False otherwise.
    """
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    files_larger_than_max_size = False

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            if size > max_size_bytes:
                log(f"Files larger than {max_size_mb}MB:{file_path}={size}", LogSeverity.ERROR, stack_trace=False)
                files_larger_than_max_size = True

    if files_larger_than_max_size:
        if raise_exception:
            raise Exception(f"Fix files larger than {max_size_mb}MB to prevent MT corrupted database.")
        else:
            log(f"Fix files larger than {max_size_mb}MB to prevent MT corrupted database.", LogSeverity.ERROR)
            return False
    else:
        return True

    # Example usage


def text_to_attribute(element: ET.Element) -> ET.Element:
    if element.text:
        # Split the text content into key-value pairs
        attributes = element.text.split('\n')
        for attribute in attributes:
            attribute = attribute.strip()
            if '=' in attribute:
                key, value = attribute.split('=', 1)
                element.attrib[key.strip()] = value.strip()
    for i in range(0, len(element)):
        element[i] = text_to_attribute(element[i])
    return element


class MT:
    # singleton: MT
    chart_original_config_file_full_path = None
    chart_config_file_full_path = None
    profiles_relative_path = path.join("MQL5", "Profiles")
    CustomProfile_chart_directory_relative_path = path.join("Charts", "CustomProfile1")
    CustomProfile_chart_config_file_name = "chart01.chr"
    CustomProfile_chart_original_config_file_name = "chart01.chr.original"
    content_of_CustomProfile_chart_config_file = """
    <chart>
    	id=0
    	symbol=CustomBTCUSDT
    	description=
    	period_type=1
    	period_size=1
    	digits=4
    	tick_size=0.000000
    	position_time=1694491200
    	scale_fix=0
    	scale_fixed_min=25460.738000
    	scale_fixed_max=26884.364000
    	scale_fix11=0
    	scale_bar=0
    	scale_bar_val=1.000000
    	scale=16
    	mode=1
    	fore=0
    	grid=1
    	volume=2
    	scroll=1
    	shift=0
    	shift_size=19.832985
    	fixed_pos=0.000000
    	ticker=1
    	ohlcv=0
    	one_click=0
    	one_click_btn=1
    	bidline=1
    	askline=0
    	lastline=0
    	days=0
    	descriptions=0
    	tradelines=1
    	tradehistory=1
    	window_left=0
    	window_top=0
    	window_right=1503
    	window_bottom=658
    	window_type=1
    	floating=0
    	floating_left=0
    	floating_top=0
    	floating_right=0
    	floating_bottom=0
    	floating_type=1
    	floating_toolbar=1
    	floating_tbstate=
    	background_color=0
    	foreground_color=16777215
    	barup_color=65280
    	bardown_color=65280
    	bullcandle_color=0
    	bearcandle_color=16777215
    	chartline_color=65280
    	volumes_color=3329330
    	grid_color=10061943
    	bidline_color=10061943
    	askline_color=255
    	lastline_color=49152
    	stops_color=255
    	windows_total=1

    	<expert>
    		name=CustomAutoLoader
    		path=Experts\CustomAutoLoader.ex5
    		expertmode=33
    		<inputs>
    		</inputs>
    	</expert>

    	<window>
    		height=100.000000
    		objects=0

    		<indicator>
    			name=Main
    			path=
    			apply=1
    			show_data=1
    			scale_inherit=0
    			scale_line=0
    			scale_line_percent=50
    			scale_line_value=0.000000
    			scale_fix_min=0
    			scale_fix_min_val=0.000000
    			scale_fix_max=0
    			scale_fix_max_val=0.000000
    			expertmode=0
    			fixed_height=-1
    		</indicator>
    	</window>
    </chart>
    """
    autoit_script_path = Path(os.getcwd(), "load_meta_trader.au3")
    autoit_executable_path = "C:\\Program Files (x86)\\AutoIt3\\AutoIt3_x64.exe"

    # CLIENT_INITIALIZED = False
    @staticmethod
    def map_symbol(symbol: str) -> str:
        return map_symbol(symbol, _meta_trader_symbol_map)

    @staticmethod
    def data_path():
        # terminal_info = mt5_client().terminal_info()
        # return terminal_info.data_path
        current_working_dir = os.getcwd()
        if not path.isfile(path.join(current_working_dir, 'MetaTrader5', 'terminal64.exe')):
            raise Exception('Install MetaTrader 5 portable under ' +
                            path.join(current_working_dir, 'MetaTrader5') + '\n' +
                            'https://drive.google.com/file/d/1YIiteGoiDxaL84ZbTCtccZi713BZssM7/view?usp=drive_link')
        return path.join(current_working_dir, 'MetaTrader5')  # , 'MQL5')

    @staticmethod
    def extract_to_data_path(source_file_path: str, make_dir: bool = False,
                             do_not_raise_exception: bool = False) -> Union[bool, None]:
        if source_file_path[-4:] != '.zip':
            raise Exception('source_file_path should end with .zip but:' + source_file_path)
        source_file_name = path.basename(source_file_path)
        source_file_extension_striped = source_file_name[:-4]
        source_file_info: FileInfoSet = extract_file_info(source_file_name)
        # destination_file_name = (map_to_meta_trader_symbol(source_file_info['symbol']) + '.' +
        #                          source_file_info['file_type'] +
        #                          '.csv')
        files_path_sub_folder = 'MQL5/Files'
        destination_folder = path.join(MT.data_path(), files_path_sub_folder)
        # destination_path = path.join(destination_folder, files_path_sub_folder) # , destination_file_name)
        if make_dir or do_not_raise_exception:
            raise Exception('Not implemented')
        with zipfile.ZipFile(source_file_path, 'r') as _zip:
            for file in _zip.filelist:
                if path.exists(path.join(destination_folder, file.filename)):
                    remove(path.join(destination_folder, file.filename))
                _destination_file = path.join(destination_folder, 'Custom' + source_file_info.symbol + '.' +
                                              source_file_info.file_type + '.csv')
                if path.exists(_destination_file):
                    remove(_destination_file)
            _zip.extract(source_file_extension_striped, destination_folder, )
            for file in _zip.filelist:
                if file.filename[:len(source_file_info.symbol)] != source_file_info.symbol:
                    _source_file = path.join(destination_folder, file.filename)
                    # _destination_file = source_file_info['symbol'] + '.' + _source_file
                    _destination_file = path.join(destination_folder, 'Custom' + source_file_info.symbol + '.' +
                                                  source_file_info.file_type + '.csv')
                    os.rename(_source_file, _destination_file)
                    log(f'The file {_destination_file} Generated! Run specific loader in MT5.', stack_trace=False)
        return

    @classmethod
    def custom_profile_chart_exists(cls):
        if cls.chart_config_file_full_path is None:
            cls.chart_config_file_full_path = (
                path.join(MT.data_path(),
                          MT.profiles_relative_path,
                          MT.CustomProfile_chart_directory_relative_path,
                          MT.CustomProfile_chart_config_file_name))
        if not path.isfile(cls.chart_config_file_full_path):
            raise Exception('Can not find CustomProfile chart config file  in ' +
                            cls.chart_config_file_full_path + '\n' +
                            'Expected to contain:' + MT.content_of_CustomProfile_chart_config_file +
                            'Add CustomSymbols to MT5, create the CustomProfile and make it to load'
                            'CustomAutoLoader under MetaTrader5/MQL5/Experts')

    @classmethod
    def custom_profile_original_chart_exists(cls):
        if cls.chart_original_config_file_full_path is None:
            cls.chart_original_config_file_full_path = (
                path.join(MT.data_path(),
                          MT.profiles_relative_path,
                          MT.CustomProfile_chart_directory_relative_path,
                          MT.CustomProfile_chart_original_config_file_name))
        if not path.isfile(cls.chart_original_config_file_full_path):
            raise Exception('Can not find CustomProfile chart original config file  in ' +
                            cls.chart_original_config_file_full_path + '\n' +
                            'Expected to contain:' + cls.chart_original_config_file_full_path +
                            'Add CustomSymbols to MT5, create the CustomProfile and make it to load'
                            'CustomAutoLoader under MetaTrader5/MQL5/Experts')

    @classmethod
    def autoit_exists(cls):
        if not path.isfile(cls.autoit_executable_path):
            raise Exception('Install AutoIt3 in ' + cls.autoit_executable_path)

        # Specify the path to your AutoIt script

    @classmethod
    def rate_load_requirements(cls):
        cls.find_conflicting_profiles()
        cls.custom_profile_chart_exists()
        cls.custom_profile_original_chart_exists()
        cls.check_custom_profile_chart_content()
        cls.autoit_exists()
        cls.autoit_runner_script_exists()
        check_file_sizes()

    @classmethod
    def load_rates(cls):
        cls.rate_load_requirements()
        cls.kill_meta_trader()
        # Use subprocess to run the script
        result = subprocess.run([cls.autoit_executable_path, cls.autoit_script_path], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Print the result
        log("STDOUT:" + result.stdout.decode('utf-8'), stack_trace=False)
        log("STDERR:" + result.stderr.decode('utf-8'), stack_trace=False)
        time.sleep(2)
        cls.kill_meta_trader()
        time.sleep(2)

    @classmethod
    def kill_meta_trader(cls):
        # Define the process name for MetaTrader 5
        process_names = ["terminal.exe", "terminal64.exe"]
        # Check if MetaTrader 5 is running
        for process in psutil.process_iter(attrs=['pid', 'name']):
            if process.info['name'] in process_names:
                log(f"Found MetaTrader 5 running with PID: {process.pid}", stack_trace=False)
                # Terminate the MetaTrader 5 process
                try:
                    psutil.Process(process.pid).terminate()
                    log(f"MetaTrader 5 terminated successfully (pid={process.pid}).", stack_trace=False)
                except psutil.NoSuchProcess:
                    log(f"MetaTrader 5 process not found (pid={process.pid}).", stack_trace=False)

    @classmethod
    def check_custom_profile_chart_content(cls, debug: bool = False):
        # Parse the content of chart01.chr and chart01.chr.original as XML
        with open(cls.chart_config_file_full_path, 'r', encoding='utf-16-le') as chart_file:
            chart_content = chart_file.read()

        with open(cls.chart_original_config_file_full_path, 'r', encoding='utf-16-le') as original_file:
            original_content = original_file.read()

        # Parse the data as XML
        chart_tree = ET.ElementTree(ET.fromstring(chart_content))
        original_tree = ET.ElementTree(ET.fromstring(original_content))

        # Get the root elements
        chart_root: ET.Element = chart_tree.getroot()
        original_root: ET.Element = original_tree.getroot()

        # convert texts to attributes
        chart_root = text_to_attribute(chart_root)
        original_root = text_to_attribute(original_root)

        # Compare the XML elements recursively and log differences
        if not compare_tree(chart_root, original_root):
            raise Exception(f'CustomProfile({cls.chart_config_file_full_path})and it\'s '
                            f'backup({cls.chart_original_config_file_full_path}) are not the same')

    @classmethod
    def find_conflicting_profiles(cls, profiles_directory_path: str = None) -> None:
        """
        List all directories under MQL5/Profiles/Charts/ and if there is more than one directory name starts with C
        warn as: "More than one profile starts with C which may conflict with CustomProfile selection. List of
        conflicting folders:" and list all conflicting profile name.
        """
        if profiles_directory_path is None or len(profiles_directory_path) == 0:
            profiles_directory_path = path.join(cls.data_path(), cls.profiles_relative_path, 'Charts')

        # List all directories in the specified folder
        directories = [d for d in os.listdir(profiles_directory_path) if
                       os.path.isdir(os.path.join(profiles_directory_path, d))]

        # Check for conflicting profile names starting with 'C'
        profile_names_with_c = [d for d in directories if d.upper().startswith('C')]

        if len(profile_names_with_c) > 1:
            raise Exception("More than one profile starts with C which may conflict with CustomProfile selection. "
                            f"List of conflicting folders:[{', '.join(profile_names_with_c)}]")

    @classmethod
    def autoit_runner_script_exists(cls):
        if not path.isfile(cls.autoit_script_path):
            raise Exception(cls.autoit_script_path + ' does not exist!')


# def zz_write_hst(df: pt.DataFrame[OHLCV], symbol: str, period: int, price_precision: int, filepath: str):
#     # Define the structures as per the MT4 HST format
#     HST_HEADER_FORMAT = '64s12si2i13i'
#     HST_RECORD_FORMAT = 'I5d'
#     # Construct the header data
#     header = struct.pack(HST_HEADER_FORMAT,
#                          b'MetaTrader 4 Copyright',  # copyright
#                          symbol.encode('ascii'),  # symbol
#                          period,  # period
#                          price_precision,  # digits
#                          int(time.time()),  # timesign
#                          int(time.time()),  # last_sync
#                          *([0] * 12)  # unused
#                          )
#     with open(filepath, 'wb') as f:
#         f.write(header)
#
#         for index, row in df.iterrows():
#             record = struct.pack(HST_RECORD_FORMAT,
#                                  int(index.timestamp()),
#                                  row['open'],
#                                  row['low'],
#                                  row['high'],
#                                  row['close'],
#                                  row['volume']
#                                  )
#             f.write(record)
