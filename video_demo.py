import cv2
from ultralytics import YOLO

# --- 1. 加载模型 ---
# 确保你有这个权重文件，或者换成你自己的路径
model = YOLO("./weights/yolov8s.pt")

# --- 2. 定义视频源和输出路径 ---
input_video_path = "2025-10-02 202346.mp4"
output_video_path = "out_video.mp4"  # 输出的视频文件名

# --- 3. 打开视频文件并获取其属性 ---
# 使用 OpenCV 打开视频
cap = cv2.VideoCapture(input_video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"Error: Could not open video file '{input_video_path}'.")
    exit()

# 获取视频的宽度、高度和帧率(fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# --- 4. 设置视频编写器 (VideoWriter) ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 创建 VideoWriter 对象, 参数: (输出文件名, 编解码器, 帧率, (宽度, 高度))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"开始处理视频，结果将保存为: {output_video_path}")

# --- 5. 逐帧处理视频 ---
while cap.isOpened():
    ret, frame = cap.read()  # 读取一帧

    if not ret:  # 如果没有读取到帧 (即视频结束)，则退出循环
        break

    # 使用 YOLO 模型进行预测
    # stream=True 对于视频处理是一个好习惯，它能提高内存效率
    results = model(frame, stream=True)

    # 遍历结果 (对于视频帧，通常只有一个结果)
    for r in results:
        # 获取绘制了检测框的图像 (numpy array)
        annotated_frame = r.plot()

        # 将带检测框的帧写入输出视频
        out.write(annotated_frame)

        # (可选) 在窗口中实时显示处理过程
        cv2.imshow('YOLOv8 Detection', annotated_frame)

        # 按 'q' 键可以提前退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# --- 6. 清理资源 ---
print("视频处理完成！")
cap.release()  # 释放视频捕获对象
out.release()  # 释放视频编写器对象
cv2.destroyAllWindows()  # 关闭所有 OpenCV 创建的窗口
