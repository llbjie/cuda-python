import time
import numpy as np
import cupy as cp
from cuda.core.experimental import Device, LegacyPinnedMemoryResource

MEMCOPY_ITERATIONS = 50

class MemoryMode:
    PAGEABLE = 0
    PINNED = 1
class BandwidthTester:
    def __init__(self):
        self.dev = Device()
        self.dev.set_current()
        self.stream = self.dev.create_stream()
        cp.cuda.ExternalStream(int(self.stream.handle)).use()

        self.device_mr = self.dev.memory_resource
        self.pinned_mr = LegacyPinnedMemoryResource()

    def _allocate_device_buffer(self, size_bytes):
        buf = self.device_mr.allocate(size_bytes, stream=self.stream)
        arr = cp.from_dlpack(buf).view(dtype=np.uint8)
        return buf, arr

    def _allocate_pinned_host(self, size_bytes):
        buf = self.pinned_mr.allocate(size_bytes, stream=self.stream)
        arr = np.from_dlpack(buf).view(dtype=np.uint8)
        return buf, arr

    def _allocate_pageable_host(self, size_bytes):
        arr = np.empty(size_bytes, dtype=np.uint8)
        return None, arr

    def _sync(self):
        self.stream.sync()

    def _measure_repeat_copy(self, src, dst, size_bytes):
        t0 = time.perf_counter()
        for _ in range(MEMCOPY_ITERATIONS):
            src.copy_to(dst, stream=self.stream)
        self._sync()
        elapsed = time.perf_counter() - t0
        bandwidth_gbs = (size_bytes * MEMCOPY_ITERATIONS) / 1e9 / elapsed
        return bandwidth_gbs

    def test_device_to_host(self, size_bytes, mem_mode):
        if mem_mode == MemoryMode.PINNED:
            hbuf, harr = self._allocate_pinned_host(size_bytes)
        else:
            hbuf, harr = self._allocate_pageable_host(size_bytes)

        # 初始化host数据
        harr[:] = np.arange(size_bytes, dtype=np.uint8)

        dbuf, darr = self._allocate_device_buffer(size_bytes)
        darr[:] = cp.asarray(harr)  # Host -> Device 初始化设备内存

        if mem_mode == MemoryMode.PINNED:
            dst_buf, dst_arr = self._allocate_pinned_host(size_bytes)
        else:
            dst_buf, dst_arr = self._allocate_pageable_host(size_bytes)

        bandwidth = self._measure_repeat_copy(dbuf, dst_buf if dst_buf else dst_arr, size_bytes)

        # 关闭分配
        dbuf.close(self.stream)
        if hbuf:
            hbuf.close(self.stream)
        if mem_mode == MemoryMode.PINNED:
            dst_buf.close(self.stream)

        return bandwidth

    def test_host_to_device(self, size_bytes, mem_mode):
        if mem_mode == MemoryMode.PINNED:
            hbuf, harr = self._allocate_pinned_host(size_bytes)
        else:
            hbuf, harr = self._allocate_pageable_host(size_bytes)

        harr[:] = np.arange(size_bytes, dtype=np.uint8)

        dbuf, darr = self._allocate_device_buffer(size_bytes)

        bandwidth = self._measure_repeat_copy(hbuf if hbuf else harr, dbuf, size_bytes)

        dbuf.close(self.stream)
        if hbuf:
            hbuf.close(self.stream)

        return bandwidth

    def test_device_to_device(self, size_bytes):
        dbuf_src, darr_src = self._allocate_device_buffer(size_bytes)
        dbuf_dst, darr_dst = self._allocate_device_buffer(size_bytes)

        darr_src[:] = cp.arange(size_bytes, dtype=np.uint8)

        bandwidth = self._measure_repeat_copy(dbuf_src, dbuf_dst, size_bytes)
        bandwidth *= 2.0  # 双向计入

        dbuf_src.close(self.stream)
        dbuf_dst.close(self.stream)

        return bandwidth


if __name__ == "__main__":
    tester = BandwidthTester()
    size = 1024 * 1024 * 100  # 100 MB

    bw_d2h = tester.test_device_to_host(size, MemoryMode.PINNED)
    print(f"Device to Host Bandwidth (Pinned): {bw_d2h:.2f} GB/s")

    bw_h2d = tester.test_host_to_device(size, MemoryMode.PINNED)
    print(f"Host to Device Bandwidth (Pinned): {bw_h2d:.2f} GB/s")

    bw_d2d = tester.test_device_to_device(size)
    print(f"Device to Device Bandwidth: {bw_d2d:.2f} GB/s")

    # 关闭流
    tester.stream.close()
    cp.cuda.Stream.null.use()
