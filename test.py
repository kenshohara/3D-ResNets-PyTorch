import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter


def get_video_results(outputs, class_names):
    sorted_scores, locs = torch.topk(outputs, k=min(5, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def test(data_loader, model, result_path, class_names, no_average):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    results = {'results': defaultdict(list)}
    
    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            video_ids, segments = zip(*targets)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1).cpu()

            for j in range(outputs.size(0)):
                results['results'][video_ids[j]].append({
                    'segment': segments[j],
                    'output': outputs[j]
                })

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    test_results = {'results': {}}
    if not no_average:
        for video_id, video_results in results['results'].items():
            video_outputs = [
                segment_result['output'] for segment_result in video_results
            ]
            video_outputs = torch.stack(video_outputs)
            average_scores = torch.mean(video_outputs, dim=0)
            test_results[video_id] = get_video_results(average_scores,
                                                       class_names)
    else:
        for video_id, video_results in results['results'].items():
            test_results[video_id] = []
            for segment_result in video_results:
                segment = segment_result['segment']
                result = get_video_results(segment_result['output'],
                                           class_names)
                test_results[video_id].append({
                    'segment': segment,
                    'result': result
                })

    with open(result_path, 'w') as f:
        json.dump(test_results, f)
