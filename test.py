import time
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils import AverageMeter


def test_in_buffer(model, batch_size, input_buffer, video_id_buffer,
                   output_buffer, test_results, class_names):
    while True:
        n_samples = sum([x.size(0) for x in input_buffer])
        if n_samples < batch_size:
            return input_buffer, video_id_buffer, output_buffer, test_results

        inputs, input_buffer = prepare_inputs(input_buffer, batch_size)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1)

        rest_video_id_buffer = []
        for video_id, begin_index, end_index in video_id_buffer:
            if end_index <= outputs.size(0):
                current_outputs = outputs[begin_index:end_index].cpu()
                if output_buffer:
                    current_outputs = torch.cat(
                        output_buffer + [current_outputs], dim=0)
                test_results['results'][video_id] = calculate_video_results(
                    current_outputs, video_id, class_names)
                output_buffer = []
            else:
                output_buffer.append(outputs[begin_index:].cpu())
                n_video_samples = end_index - begin_index
                rest_video_id_buffer.append([
                    video_id,
                    0,
                    n_video_samples - outputs[begin_index:].size(0),
                ])

        video_id_buffer = rest_video_id_buffer


def prepare_inputs(input_buffer, batch_size):
    n_input_samples = 0
    for buffer_index, x in enumerate(input_buffer):
        n_input_samples += x.size(0)
        if n_input_samples >= batch_size:
            n_over_samples = n_input_samples - batch_size
            break
    else:
        n_over_samples = 0

    if n_over_samples == 0:
        inputs = torch.cat(input_buffer[:(buffer_index + 1)], dim=0)
        input_buffer = input_buffer[(buffer_index + 1):]
    else:
        inputs = torch.cat(
            input_buffer[:buffer_index] +
            [input_buffer[buffer_index][:-n_over_samples]],
            dim=0)
        next_begin_index = input_buffer[buffer_index].size(0) - n_over_samples
        input_buffer = [input_buffer[buffer_index][next_begin_index:]
                       ] + input_buffer[(buffer_index + 1):]

    return inputs, input_buffer


def calculate_video_results(outputs, video_id, class_names):
    average_scores = torch.mean(outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=min(5, len(class_names)))

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return video_results


def test(data_loader, model, batch_size, result_path, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    input_buffer = []
    video_id_buffer = []
    output_buffer = []
    test_results = {'results': {}}

    with torch.no_grad():
        for i, (new_inputs, new_video_ids) in enumerate(data_loader):
            input_buffer.append(new_inputs)
            n_buffer_samples = sum([x.size(0) for x in input_buffer])
            begin_buffer_index = n_buffer_samples - new_inputs.size(0)
            end_buffer_index = n_buffer_samples
            video_id_buffer.append(
                [new_video_ids[0], begin_buffer_index, end_buffer_index])
            if n_buffer_samples < batch_size:
                continue

            data_time.update(time.time() - end_time)

            (input_buffer, video_id_buffer,
             output_buffer, test_results) = test_in_buffer(
                 model, batch_size, input_buffer, video_id_buffer,
                 output_buffer, test_results, class_names)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

        if input_buffer:
            n_buffer_samples = sum([x.size(0) for x in input_buffer])
            _, _, _, test_results = test_in_buffer(
                model, n_buffer_samples, input_buffer, video_id_buffer,
                output_buffer, test_results, class_names)

    with open(result_path, 'w') as f:
        json.dump(test_results, f)


def test_in_buffer_no_average(model, batch_size, input_buffer, target_buffer,
                              output_buffer, test_results, class_names):
    while True:
        n_samples = sum([x.size(0) for x in input_buffer])
        if n_samples < batch_size:
            return input_buffer, target_buffer, output_buffer, test_results

        inputs, input_buffer = prepare_inputs(input_buffer, batch_size)
        outputs = model(inputs)
        outputs = F.softmax(outputs, dim=1).cpu()

        for i in range(batch_size):
            video_id, segment = target_buffer[i]
            current_result = {
                'segment': segment,
                'scores': get_results(outputs[i], class_names)
            }
            test_results['results'][video_id].append(current_result)

        target_buffer = target_buffer[batch_size:]


def get_results(outputs, class_names):
    sorted_scores, locs = torch.topk(outputs, k=min(5, len(class_names)))

    results = []
    for i in range(sorted_scores.size(0)):
        results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    return results


def test_no_average(data_loader, model, batch_size, result_path, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    input_buffer = []
    target_buffer = []
    output_buffer = []
    test_results = {'results': defaultdict(list)}

    with torch.no_grad():
        for i, (new_inputs, new_targets) in enumerate(data_loader):
            input_buffer.append(new_inputs)
            n_buffer_samples = sum([x.size(0) for x in input_buffer])
            begin_buffer_index = n_buffer_samples - new_inputs.size(0)
            end_buffer_index = n_buffer_samples
            target_buffer.extend(new_targets)
            if n_buffer_samples < batch_size:
                continue

            data_time.update(time.time() - end_time)

            (input_buffer, target_buffer,
             output_buffer, test_results) = test_in_buffer_no_average(
                 model, batch_size, input_buffer, target_buffer, output_buffer,
                 test_results, class_names)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

        if input_buffer:
            n_buffer_samples = sum([x.size(0) for x in input_buffer])
            _, _, _, test_results = test_in_buffer_no_average(
                model, n_buffer_samples, input_buffer, target_buffer,
                output_buffer, test_results, class_names)

    with open(result_path, 'w') as f:
        json.dump(test_results, f)