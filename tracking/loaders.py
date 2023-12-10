import cv2
import os
import time
from threading import Thread
import numpy as np

# Placeholder for the missing LOGGER object
class Logger:
    def info(self, message):
        print(f"INFO: {message}")

    def warning(self, message):
        print(f"WARNING: {message}")

LOGGER = Logger()


class LoadVideoStream:
    def __init__(self, source, fps_out=None):
        self.running = True
        self.frames = 0
        self.thread = None
        self.cap = cv2.VideoCapture(source)
        self.imgs = []

        # Assert on source Path
        if not os.path.exists(source):
            raise ValueError(f'ERROR ❌ Source {source} does not exist')
        self.source = str(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        self.vid_stride = 1 if fps_out is None else int(np.ceil(self.fps / fps_out))

        self.thread = Thread(target=self.update, daemon=True)
        LOGGER.info(f'Success ✅ ({self.frames} frames at {self.fps:.2f} FPS)')
        self.thread.start()

    def update(self):
        n = 0
        while self.running and self.cap.isOpened() and n < (self.frames - 1):
            n += 1
            self.cap.grab()
            if n % self.vid_stride == 0:
                success, im = self.cap.retrieve()
                if not success:
                    im = np.zeros((0, 0, 3), dtype=np.uint8)
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.cap.open(self.source)
                self.imgs.append(im)

    def close(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        try:
            self.cap.release()
        except Exception as e:
            LOGGER.warning(f'WARNING ⚠️ Could not release VideoCapture object: {e}')
        cv2.destroyAllWindows()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        images = []
        while not self.imgs:
            if not self.thread.is_alive():
                self.close()
                raise StopIteration
            time.sleep(1 / self.fps)
            if not self.imgs:
                LOGGER.warning(f'WARNING ⚠️ Waiting for stream {self.source}')

        images.append(self.imgs.pop(0))
        return self.source, images, None, ''

    def __len__(self):
        return 1
