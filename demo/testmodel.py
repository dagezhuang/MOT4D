import os
import pandas as pd
from trackeval import TrackEval
from sort import Sort  # 假设你已经实现了 SORT 算法

# 1. 解析 VOC 格式的 XML 文件
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    frame_id = int(root.find('filename').text.split('.')[0])
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append({
            'class': class_name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    return frame_id, objects

# 2. 保存跟踪结果
def save_tracking_results(track_results, output_file):
    with open(output_file, 'w') as f:
        for frame_id, frame_tracks in track_results.items():
            for track in frame_tracks:
                x, y, w, h = track['bbox']
                track_id = track['id']
                f.write(f"{frame_id},{track_id},{x},{y},{w},{h},1,-1,-1,-1\n")

# 3. 主流程
xml_dir = 'path/to/your/xml/files'
output_file = 'tracked_results.txt'
track_results = {}

# 初始化 SORT 跟踪器
sort_tracker = Sort()

# 遍历每一帧的 XML 文件
xml_files = sorted(os.listdir(xml_dir))
for xml_file in xml_files:
    frame_id, detections = parse_voc_xml(os.path.join(xml_dir, xml_file))
    detections = [obj['bbox'] for obj in detections]  # 提取边界框
    track_results[frame_id] = sort_tracker.update(detections)

# 保存跟踪结果
save_tracking_results(track_results, output_file)

# 4. 解析真值文件
gt_file = 'path/to/your/gt.txt'
gt_data = parse_gt_file(gt_file)

# 5. 评估性能指标
evaluator = TrackEval()
metrics, _, _ = evaluator.evaluate(
    config={
        'GT_FOLDER': os.path.dirname(gt_file),
        'TRACKERS_FOLDER': os.path.dirname(output_file),
        'GT_FILE': os.path.basename(gt_file),
        'TRACKERS_TO_EVAL': [os.path.basename(output_file)],
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train'
    }
)

# 打印评估结果
print(metrics)
# import os
# import shutil

# def organize_xml_files(xml_folder,video_floder):
#     """
#     遍历源文件夹中的 XML 文件，根据文件名的前两位序号将文件拷贝到对应的视频文件夹中，
#     并将文件名重命名为6位数字的 XML 文件。
#     :param source_folder: 包含原始 XML 文件的文件夹路径
#     """
#     # 遍历源文件夹中的所有文件
#     for filename in os.listdir(xml_folder):
#         if filename.endswith('.xml'):
#             # 提取文件名中的视频序号和帧序号
#             video_id = filename[:2]  # 前两位表示视频序号
#             frame_id = filename[2:8]  # 中间六位表示帧序号

            
#             # 创建目标文件夹路径
#             video_folder1 = video_folder +'/video'+str(int(video_id))
#             video_folder2 = video_folder1 +'/det/'
#             if not os.path.exists(video_folder2):
#                 os.makedirs(video_folder2)  # 如果目标文件夹不存在，则创建

#             # 创建目标文件路径
#             target_file = os.path.join(video_folder2, f"{frame_id}.xml")

#             print(target_file)

#             # 源文件路径
#             source_file = os.path.join(xml_folder, filename)
#             print(source_file)

#             # 拷贝并重命名文件
#             shutil.copy(source_file, target_file)
#             print(f"Copied {filename} to {target_file}")

#     print("File organization complete.")

# # 使用示例
# xml_folder = '/home/ubuntu/mmdetection4/demo/FLY_det_result/' # 替换为你的源文件夹路径
# video_folder = '/home/ubuntu/PFTrack/data/VTMOT/test'
# organize_xml_files(xml_folder,video_folder)