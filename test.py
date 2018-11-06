import torch
import torch.nn.functional as F
import time
import os
import sys
import json

from utils import AverageMeter


def prepare_inputs(input_buffer, batch_size):
    n_input_samples = 0
    for buffer_index, x in enumerate(input_buffer):
        n_input_samples += x.size(0)
        if n_input_samples >= batch_size:
            n_over_samples = n_input_samples - batch_size
            break

    if buffer_index == 0:
        inputs = input_buffer[buffer_index][:batch_size]
        next_begin_index = batch_size
    else:
        inputs = torch.cat(
            [input_buffer[i] for i in range(buffer_index)] +
            [input_buffer[buffer_index][:-n_over_samples]],
            dim=0)
        next_begin_index = input_buffer[buffer_index].size(0) - n_over_samples

    if n_over_samples > 0:
        input_buffer = [input_buffer[buffer_index][next_begin_index:]
                       ] + input_buffer[(buffer_index + 1):]
    else:
        input_buffer = input_buffer[(buffer_index + 1):]

    return inputs, input_buffer


def calculate_video_results(outputs, video_id, test_results, class_names):
    average_scores = torch.mean(outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=5)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    input_buffer = []
    video_id_buffer = []
    n_samples = 0
    output_buffer = []
    test_results = {'results': {}}

    with torch.no_grad():
        for i, (new_inputs, new_video_ids) in enumerate(data_loader):
            n_prev_buffer_samples = sum([x.size(0) for x in input_buffer])
            n_samples = n_prev_buffer_samples + new_inputs.size(0)
            video_id_buffer.append(
                [new_video_ids[0], n_prev_buffer_samples, n_samples])
            input_buffer.append(new_inputs)
            if n_samples < opt.batch_size:
                continue

            data_time.update(time.time() - end_time)

            while True:
                inputs, input_buffer = prepare_inputs(input_buffer,
                                                      opt.batch_size)
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)

                for video_id_index, (video_id, begin_index,
                                     end_index) in enumerate(video_id_buffer):
                    if end_index <= outputs.size(0):
                        current_outputs = outputs[begin_index:end_index].cpu()
                        if output_buffer:
                            current_outputs = torch.cat(
                                output_buffer + [current_outputs], dim=0)
                        calculate_video_results(current_outputs, video_id,
                                                test_results, class_names)
                        output_buffer = []
                    else:
                        output_buffer.append(outputs[begin_index:].cpu())
                        n_video_samples = end_index - begin_index
                        video_id_buffer = [[
                            video_id,
                            0,
                            n_video_samples - outputs[begin_index:].size(0),
                        ]]
                        break
                else:
                    video_id_buffer = video_id_buffer[(video_id_index + 1):]

                n_samples -= opt.batch_size
                if n_samples < opt.batch_size:
                    break

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time))

    with open(opt.result_path / '{}.json'.format(opt.test_subset), 'w') as f:
        json.dump(test_results, f)
