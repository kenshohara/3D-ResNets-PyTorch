from __future__ import print_function, division
import os
import sys
import subprocess


if __name__=="__main__":
  dir_path = sys.argv[1]
  dst_dir_path = sys.argv[2]

  for file_name in os.listdir(dir_path):
    if '.mp4' not in file_name:
      continue
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_dir_path, name)

    video_file_path = os.path.join(dir_path, file_name)
    p = subprocess.Popen('ffprobe {}'.format(video_file_path),
                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, res = p.communicate()
    res = res.decode('utf-8')

    duration_index = res.find('Duration:')
    duration_str = res[(duration_index + 10):(duration_index + 21)]
    hour = float(duration_str[0:2])
    minute = float(duration_str[3:5])
    sec = float(duration_str[6:10])
    total_sec = hour * 3600 + minute * 60 + sec

    n_frames = len(os.listdir(dst_directory_path))
    if os.path.exists(os.path.join(dst_directory_path, 'fps')):
      n_frames -= 1

    fps = round(n_frames / total_sec, 2)

    print(video_file_path, os.path.exists(video_file_path), fps)
    with open(os.path.join(dst_directory_path, 'fps'), 'w') as fps_file:
      fps_file.write('{}\n'.format(fps))
