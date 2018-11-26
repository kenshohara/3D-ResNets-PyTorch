import sys
import subprocess
from pathlib import Path


def class_process(class_dir_path, dst_root_path, fps=-1):
    if not class_dir_path.is_dir():
        return

    dst_class_path = dst_root_path / class_dir_path.name
    dst_class_path.mkdir(exist_ok=True)

    for video_file_path in sorted(class_dir_path.iterdir()):
        if '.avi' not in video_file_path.name:
            continue
        name = video_file_path.stem
        dst_dir_path = dst_class_path / name

        if dst_dir_path.exists():
            continue

        p = subprocess.Popen(
            'ffprobe -hide_banner -show_entries stream=width,height "{}"'.
            format(video_file_path),
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        res = p.communicate()[0].decode('utf-8').split('\n')
        if len(res) <= 3:
            continue
        width = int([x.split('=') for x in res if 'width' in x][0][1])
        height = int([x.split('=') for x in res if 'height' in x][0][1])

        if width > height:
            scale_param = '-1:240'
        else:
            scale_param = '240:-1'

        fps_param = ''
        if fps > 0:
            fps_param = ',fps={}'.format(fps)

        cmd = 'ffmpeg -i \"{}\" -vf "scale={}{}" \"{}/image_%05d.jpg\"'.format(
            video_file_path, scale_param, fps_param, dst_dir_path)

        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == '__main__':
    dir_path = Path(sys.argv[1])
    dst_dir_path = Path(sys.argv[2])
    fps = -1
    if len(sys.argv) > 3:
        fps = int(sys.argv[3])

    for class_dir_path in sorted(dir_path.iterdir()):
        class_process(class_dir_path, dst_dir_path, fps)
