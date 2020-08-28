"""
A basic example of how to use threading to generate images.
This gives only a small improvement and I'm still trying
to figure out why.
"""

import queue
import threading
from mandelpy import create_image, Settings, power
from PIL import ImageFilter

images_folder = r"..\images\increasing_powers5"
video_file = r"..\test5.mp4"

# The queue for tasks
q = queue.Queue()


# Worker, handles each task
def worker():
    while True:
        i = q.get()
        if i is None:
            break
        n = 1 + i/50
        p = (n - 2) * abs(n - 2) + 2

        settings = Settings(transform=lambda z: power(z, p) + 1.5 * z - 0.5 - 0.25j,
                            width=2000, height=2000,
                            block_size=(1000, 1000),
                            mirror_x=False)
        img = create_image(settings, verbose=True)
        img = img.filter(ImageFilter.GaussianBlur(1))
        img = img.resize((1920, 1080))
        img.save(rf"{images_folder}\Pic{i}.jpg", optimize=True, quality=90)

        q.task_done()


def start_workers(worker_pool=1000):
    threads = []
    for i in range(worker_pool):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
    return threads


def stop_workers(threads):
    # stop workers
    for i in threads:
        q.put(None)
    for t in threads:
        t.join()


def create_queue(task_items):
    for item in task_items:
        q.put(item)


if __name__ == "__main__":
    # Dummy tasks
    tasks = [item for item in range(100)]

    # Start up your workers
    workers = start_workers(worker_pool=2)
    create_queue(tasks)

    # Blocks until all tasks are complete
    q.join()

    stop_workers(workers)