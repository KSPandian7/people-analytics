import time
import numpy as np
from openvino.runtime import Core

from config import *

def main():
    core = Core()

    model = core.read_model("peta_attributes_openvino.xml")
    compiled_model = core.compile_model(model, "CPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    # Dummy input
    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # Warm-up
    for _ in range(10):
        compiled_model([dummy_input])

    # Benchmark
    runs = 100
    start = time.time()

    for _ in range(runs):
        compiled_model([dummy_input])

    end = time.time()

    avg_latency = (end - start) / runs * 1000
    fps = 1000 / avg_latency

    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")

if __name__ == "__main__":
    main()
