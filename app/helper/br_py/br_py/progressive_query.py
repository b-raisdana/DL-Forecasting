# import psutil
# from clickhouse_async import Client
#
# _cached_num_threads = None
#
# """
# import os
#
# workers = (2 * psutil.cpu_count(logical=True))) + 1
# gunicorn --workers=$(($(nproc) * 2 + 1)) myapp:app
# """
#
#
# async def num_of_threads():
#     global _cached_num_threads
#     if _cached_num_threads is None:
#         process = psutil.Process()
#         _cached_num_threads = process.num_threads()
#     return _cached_num_threads
#
#
# async def execute_query_with_progress(query, max_threads: float | int = 0.33):
#     assert max_threads > 0
#
#     client = Client(host='localhost', user='default', password='', database='default')
#     num_threads = await num_of_threads()
#     if isinstance(max_threads, float):
#         max_threads = int(num_threads * max_threads)
#     print(f"Max Threads for ClickHouse query: {max_threads}")
#
#     async def handle_progress_or_result(result):
#         if isinstance(result, dict):
#             rows_read = result.get('read_rows', None)
#             total_rows = result.get('total_rows', None)
#
#             if rows_read is not None and total_rows not in (None, 0):
#                 percent_complete = (rows_read / total_rows) * 100
#             else:
#                 percent_complete = None
#
#             yield {"progress": percent_complete, "rows_read": rows_read, "total_rows": total_rows}
#         else:
#             yield {"result": result}
#
#     for result in client.execute_iter(query, with_column_types=True):
#         async for progress_or_result in handle_progress_or_result(result):
#             yield progress_or_result