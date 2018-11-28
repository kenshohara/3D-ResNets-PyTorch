import subprocess
import argparse
from pathlib import Path

from joblib import Parallel, delayed


def video_process(video_file_path, dst_root_path, fps=-1):
    if '.mp4' not in video_file_path.name:
        return
    name = video_file_path.stem
    dst_dir_path = dst_root_path / name

    if dst_dir_path.exists():
        return
    else:
        dst_dir_path.mkdir()

    p = subprocess.Popen(
        'ffprobe -hide_banner -show_entries stream=width,height "{}"'.format(
            video_file_path),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    res = p.communicate()[0].decode('utf-8').split('\n')
    if len(res) <= 3:
        return
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dir_path', default=None, type=Path, help='Directory path of videos')
    parser.add_argument(
        'dst_path',
        default=None,
        type=Path,
        help='Directory path of jpg videos')
    parser.add_argument(
        '--n_jobs', default=-1, type=int, help='Number of parallel jobs')
    parser.add_argument(
        '--fps',
        default=-1,
        type=int,
        help=('Frame rates of output videos. '
              '-1 means original frame rates.'))
    args = parser.parse_args()

    video_file_paths = [x for x in sorted(args.dir_path.iterdir())]
    status_list = Parallel(n_jobs=args.n_jobs)(
        delayed(video_process)(video_file_path, args.dst_path, args.fps)
        for video_file_path in video_file_paths)
