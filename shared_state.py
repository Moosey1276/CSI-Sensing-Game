import threading

stop_event = threading.Event()

class SharedPose:
    def __init__(self):
        self.lock = threading.Lock()
        self.pose = "unknown"

    def set_pose(self, pose):
        with self.lock:
            self.pose = pose

    def get_pose(self):
        with self.lock:
            return self.pose

shared_pose = SharedPose()
