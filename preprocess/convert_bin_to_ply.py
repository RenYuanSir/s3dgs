"""
Script to convert COLMAP points3D.bin to points3D.ply
Reference: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py
"""

import os
import struct
import numpy as np
import plyfile
import argparse

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
    """Read and unpack bytes from a binary file."""
    bytes_data = fid.read(num_bytes)
    # 增加一个校验，防止读取到文件末尾导致的不完整数据报错
    if len(bytes_data) < num_bytes:
        raise ValueError(f"Expected {num_bytes} bytes but only got {len(bytes_data)}")
    return struct.unpack(endian_char + format_char_sequence, bytes_data)

def convert_points3D_bin_to_ply(bin_path, ply_path):
    print(f"Converting {bin_path} -> {ply_path} ...")

    xyzs = []
    rgbs = []

    with open(bin_path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"Found {num_points} points.")

        for _ in range(num_points):
            # Binary format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            # FIXED HERE: 8 + 3*8 + 3*1 + 8 = 43 bytes
            binary_chunk = read_next_bytes(fid, 43, "QdddBBBd")

            # Extract XYZ and RGB
            xyz = binary_chunk[1:4]
            rgb = binary_chunk[4:7]

            xyzs.append(xyz)
            rgbs.append(rgb)

            # Read Track length
            track_length = read_next_bytes(fid, 8, "Q")[0]

            # Skip Track content (2 * uint32 * length)
            # image_id(4) + point2D_idx(4) = 8 bytes per track element
            fid.seek(8 * track_length, 1)

    # Convert to numpy
    xyzs = np.array(xyzs, dtype=np.float32)
    rgbs = np.array(rgbs, dtype=np.uint8)

    # Create PLY structure
    vertex_data = [
        (xyzs[i, 0], xyzs[i, 1], xyzs[i, 2], rgbs[i, 0], rgbs[i, 1], rgbs[i, 2])
        for i in range(len(xyzs))
    ]

    vertex_elements = np.array(vertex_data, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    el = plyfile.PlyElement.describe(vertex_elements, 'vertex')
    ply_data = plyfile.PlyData([el], text=False) # Binary PLY
    ply_data.write(ply_path)

    print(f"Success! Saved to {ply_path}")

if __name__ == "__main__":
    # 配置你的路径
    INPUT_BIN = r"D:\PythonProject\PythonProject\data\video_data\colmap_data\video3_output\sparse\0\points3D.bin"
    OUTPUT_PLY = r"D:\PythonProject\PythonProject\data\video_data\colmap_data\video3_output\sparse\0\points3D.ply"

    if not os.path.exists(INPUT_BIN):
        print(f"Error: {INPUT_BIN} not found.")
    else:
        convert_points3D_bin_to_ply(INPUT_BIN, OUTPUT_PLY)