"""Tests for bounded synchronous inference capacity."""
import threading
import time
import unittest

from remote_inference.capacity import BoundedWorkQueue


class BoundedWorkQueueTest(unittest.TestCase):
    def test_serializes_work_and_rejects_beyond_capacity(self):
        queue = BoundedWorkQueue(concurrency=1, max_in_flight=2)
        first_started = threading.Event()
        release_first = threading.Event()
        accepted = []

        def run_first():
            with queue.reserve() as reserved:
                accepted.append(reserved)
                first_started.set()
                release_first.wait(1)

        def run_second():
            with queue.reserve() as reserved:
                accepted.append(reserved)

        first = threading.Thread(target=run_first)
        second = threading.Thread(target=run_second)
        first.start()
        self.assertTrue(first_started.wait(1))
        second.start()
        time.sleep(0.05)

        with queue.reserve() as reserved:
            self.assertFalse(reserved)

        release_first.set()
        first.join(1)
        second.join(1)
        self.assertEqual(accepted, [True, True])

    def test_validates_limits(self):
        with self.assertRaises(ValueError):
            BoundedWorkQueue(concurrency=0, max_in_flight=1)
        with self.assertRaises(ValueError):
            BoundedWorkQueue(concurrency=2, max_in_flight=1)


if __name__ == "__main__":
    unittest.main()
