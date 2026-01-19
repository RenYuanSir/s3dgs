import cv2
import os
import argparse


def extract_frames(video_path, output_folder, interval=0.1):
    """
    从视频中每隔指定秒数截取一帧并保存

    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹路径
        interval: 截帧间隔（秒），默认0.1秒
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"视频信息:")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f} 秒")

    # 计算每隔多少帧截取一次
    frame_interval = int(fps * interval)

    if frame_interval < 1:
        frame_interval = 1
        print(f"警告: 间隔太小，调整为每帧截取")

    print(f"截帧设置:")
    print(f"  时间间隔: {interval} 秒")
    print(f"  帧间隔: {frame_interval} 帧")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 每隔指定帧数保存一次
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            output_path = os.path.join(
                output_folder,
                f"frame_{timestamp:07.3f}.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"已保存 {saved_count} 帧...")

        frame_count += 1

    cap.release()
    print(f"\n完成! 共保存 {saved_count} 帧到 {output_folder}")


def main():
    parser = argparse.ArgumentParser(description='从视频中截取帧')
    parser.add_argument('video_path', type=str, help='视频文件路径')
    parser.add_argument('output_folder', type=str, help='输出文件夹路径')
    parser.add_argument(
        '--interval',
        type=float,
        default=0.1,
        help='截帧间隔（秒），默认0.1秒'
    )

    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.interval)


if __name__ == "__main__":
    # 示例用法（也可以直接修改这里的参数运行）
    # extract_frames("input.mp4", "output_frames", interval=0.1)

    main()
