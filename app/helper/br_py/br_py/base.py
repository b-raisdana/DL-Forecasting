import asyncio

from .do_log.log_it import init_logger
from .profiling.base import init_global_profile_to_db

#

async def br_lib_init(path_of_logs: str, log_to_std_out_level: int, log_to_file_level: int,
                      root_path: str, global_profile_to_db: bool = False, file_log_rotation_size = "100 MB", file_log_retention_duration = "30 days"):
    init_logger(path_of_logs=path_of_logs, log_to_std_out_level=log_to_std_out_level,
                log_to_file_level=log_to_file_level, root_path=root_path,
                file_log_rotation_size = file_log_rotation_size,
                file_log_retention_duration = file_log_retention_duration)
    if global_profile_to_db:
        init_global_profile_to_db()


def sync_br_lib_init(path_of_logs: str, log_to_std_out_level: int, log_to_file_level: int,
                     root_path: str, global_profile_to_db: bool = False):
    asyncio.run(br_lib_init(path_of_logs=path_of_logs, root_path=root_path, log_to_file_level=log_to_file_level,
                            log_to_std_out_level=log_to_std_out_level))

