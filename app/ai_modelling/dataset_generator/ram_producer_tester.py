# monitor_meta_queue.py
"""
Connect to the multiprocessing.Manager started by the producer program
and print the current size of the shared meta-queue every second.

Run your producer/server first (the one that starts MyManager on 127.0.0.1:50055
with authkey b"secret123"), then start this script.
"""

import time
from multiprocessing.managers import SyncManager


# ─── Manager subclass with the same registration name ─────────────────
class MyManager(SyncManager):
    pass


# Must use exactly the same name the server registered
MyManager.register("get_meta_queue")


def main():
    # ── address and authkey must match the server ─────────────────────
    mgr = MyManager(address=("127.0.0.1", 50055), authkey=b"secret123")
    mgr.connect()                         # dial the manager
    meta_q = mgr.get_meta_queue()         # proxy object

    print("Connected to meta_queue.")
    while True:
        try:
            size = meta_q.qsize()         # remote call
            print(f"meta_queue size = {size}")
        except Exception as e:
            print("Error while querying queue:", e)
        time.sleep(1)


if __name__ == "__main__":
    main()
