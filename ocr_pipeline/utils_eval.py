import os
import statistics
import time

import Levenshtein  # pip install python-Levenshtein
import pandas as pd
import torch
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def recognition_metrics(predictions, labels, vis_res=True, file_name="results.csv", verbose=True):

    all_num = 0
    correct_num = 0
    norm_edit_dis = 0.0
    total_edit_dist = 0.0
    total_length = 0

    if vis_res:
        res_dict = {'index':[], 'label':[], 'pred':[], 'edit_dist':[], 'label_len':[]}

    for pred, label in zip(predictions, labels):
        # pred = pred.replace(" ", "")
        # target = target.replace(" ", "")
        edit_dist = Levenshtein.distance(pred, label)
        max_len = max(len(pred), len(label), 1)

        norm_edit_dis += edit_dist / max_len

        total_edit_dist += edit_dist
        total_length += max_len

        if edit_dist == 0:
            correct_num += 1
        all_num += 1

        if vis_res:
            res_dict['index'].append(all_num)
            res_dict['label'].append(label)
            res_dict['pred'].append(pred)
            res_dict['edit_dist'].append(edit_dist)
            res_dict['label_len'].append(len(label))

    if vis_res:
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv("./results/" + file_name, mode='w', index=False, header=True)

    results = {
        'abs_match': correct_num,
        'wrr': 100 * correct_num / all_num,
        'total_ned': norm_edit_dis,
        'crr': 100 * (1 - total_edit_dist / total_length)
    }

    if verbose:
        print(f"Absolute Word Match Count: {results['abs_match']}")
        print(f"Word Recognition Rate (WRR)%: {results['wrr']:.2f}")
        print(f"Total Normal Edit Distance (NED): {results['total_ned']:.4f}")
        print(f"Character Recognition Rate (CRR)%: {results['crr']:.2f}")
        print()

    return results


def recognition_metrics_atlength(predictions, labels):

    # print(len(labels))
    # lengths = [len(label) for label in labels]
    # lengths = statistics.quantiles(lengths, n=4)
    # lengths.extend([0,100])
    # lengths = sorted(lengths)
    # print(lengths)
    # lengths = [0, 5, 10, 15, 100]
    lengths = [0, 4, 8, 12, 100]

    for i in range(len(lengths)-1):

        all_num = 0
        correct_num = 0
        norm_edit_dis = 0.0
        total_edit_dist = 0.0
        total_length = 0

        for pred, label in zip(predictions, labels):

            if len(label) > lengths[i] and len(label) <= lengths[i+1]:
                # pred = pred.replace(" ", "")
                # target = target.replace(" ", "")
                edit_dist = Levenshtein.distance(pred, label)
                max_len = max(len(pred), len(label), 1)

                norm_edit_dis += edit_dist / max_len

                total_edit_dist += edit_dist
                total_length += max_len

                if edit_dist == 0:
                    correct_num += 1
                all_num += 1

        wrr = 0 if all_num == 0 else correct_num / all_num
        crr = 0 if total_length == 0 else 1 - total_edit_dist / total_length
        results = {
            'long_words': all_num,
            'abs_match': correct_num,
            'wrr': 100*wrr,
            'total_ned': norm_edit_dis,
            'crr': 100*crr
        }

        print(f"At length {lengths[i]+1} to {lengths[i+1]}")
        print(f"Word Count: {results['long_words']}")
        # print(f"Absolute Word Match Count: {results['abs_match']}")
        print(f"Word Recognition Rate (WRR)%: {results['wrr']:.2f}")
        # print(f"Total Normal Edit Distance (NED): {results['total_ned']:.4f}")
        # print(f"Character Recognition Rate (CRR)%: {results['crr']:.2f}")
    print()


def mispredicted_conjunct_count(predictions, labels, verbose=True):

    total_conjuncts = 0
    wrong_conjuncts = 0

    conjunct_list = ['ল্ম', 'ক্ষ', 'দ্ব', 'ক্ত', 'ত্ত', 'চ্ছ', 'ন্ট', 'স্ত', 'ত্ম', 'ন্দ', 'ন্ত', 'ন্ন', 'স্ট', 'ঙ্ক',
                     'ন্ড', 'স্ব', 'ফ্ল', 'চ্চ', 'ষ্প', 'ত্ন', 'দ্দ', 'ম্প', 'ণ্ড', 'ল্ড', 'ক্স', 'ন্স', 'ঞ্জ', 'ক্ল',
                     'ম্ব', 'ন্ধ', 'ল্প', 'ণ্ট', 'স্প', 'ঙ্গ', 'স্ন', 'শ্ব', 'গ্ল', 'ত্ত্ব', 'স্ক', 'প্ত', 'ষ্ট', 'স্ফ',
                     'দ্ধ', 'ন্থ', 'স্থ', 'ঞ্চ', 'ষ্ঠ', 'শ্চ', 'ল্ল', 'ল্ক', 'শ্ল', 'ত্ব', 'প্ট', 'ব্দ', 'ঙ্ক্ষ', 'স্ম',
                     'শ্ন', 'জ্জ', 'জ্ঞ', 'দ্গ', 'ল্ট', 'ম্ন', 'ম্ম', 'প্ল', 'ক্ট', 'ম্ভ', 'ব্ল', 'ন্ব', 'দ্ভ', 'ট্ট', 
                     'ক্ব', 'প্প', 'ক্ক', 'ষ্ক', 'গ্ন', 'হ্ম', 'ম্ল', 'প্ন', 'ব্ব', 'দ্ম', 'ক্ষ্ম', 'ষ্ণ', 'জ্ব', 'স্ল', 'শ্ম',
                     'ন্জ', 'ল্ফ', 'ঙ্ঘ', 'ব্জ', 'ঞ্ঝ', 'ন্ম', 'ক্ষ্ণ', 'ধ্ব', 'ফ্ট', 'স্প্ল', 'গ্ব', 'প্স', 'ষ্ম', 'গ্ম',
                     'ঙ্খ', 'ম্ফ', 'ব্ধ', 'হ্ন', 'ড্ড', 'ল্স', 'ত্থ', 'জ্জ্ব', 'হ্ণ', 'ন্ঠ', 'গ্ধ', 'হ্ব', 'ণ্ঠ', 'ণ্ব', 'দ্ধ্ব',
                     'ল্গ', 'ক্ন', 'ঘ্ন', 'গ্গ', 'হ্ল', 'ণ্ণ', 'ষ্ব', 'ন্দ্ব', 'দ্ঘ', 'ঞ্ছ', 'স্খ', 'ঙ্ম', 'ক্ম', 'ল্ব', 'ম্প্ল',
                     'থ্ব', 'শ্ত', 'ন্ত্ব', 'ণ্ঢ', 'শ্ছ', 'ষ্ফ', 'চ্ছ্ব', 'দ্দ্ব', 'স্ত্ব', 'ণ্ম', 'জ্ঝ', 'ধ্ন', 'ট্ব', 'চ্ঞ', 'ব্ভ']


    for pred, label in zip(predictions, labels):
        # pred = pred.replace(" ", "")
        # target = target.replace(" ", "")
        gt_conjuncts = []
        for conjunct in conjunct_list:
            if conjunct in label:
                gt_conjuncts.append(conjunct)
                total_conjuncts += 1

        temp_pred = pred
        for conjunct in gt_conjuncts:

            if conjunct not in temp_pred:
                wrong_conjuncts += 1


    wrong_conjuncts_rate = wrong_conjuncts/total_conjuncts

    if verbose:
        print(f"Mispredicted Conjuncts: {wrong_conjuncts}")
        print(f"Percentage of Mispredicted Conjuncts: {wrong_conjuncts_rate:.4f}")
        print()

    return wrong_conjuncts, wrong_conjuncts_rate


def get_simple_words(predictions, labels, verbose=True):

    simple_words = 0
    simple_predictions = []
    simple_labels = []

    conjunct_list = ['ল্ম', 'ক্ষ', 'দ্ব', 'ক্ত', 'ত্ত', 'চ্ছ', 'ন্ট', 'স্ত', 'ত্ম', 'ন্দ', 'ন্ত', 'ন্ন', 'স্ট', 'ঙ্ক',
                     'ন্ড', 'স্ব', 'ফ্ল', 'চ্চ', 'ষ্প', 'ত্ন', 'দ্দ', 'ম্প', 'ণ্ড', 'ল্ড', 'ক্স', 'ন্স', 'ঞ্জ', 'ক্ল',
                     'ম্ব', 'ন্ধ', 'ল্প', 'ণ্ট', 'স্প', 'ঙ্গ', 'স্ন', 'শ্ব', 'গ্ল', 'ত্ত্ব', 'স্ক', 'প্ত', 'ষ্ট', 'স্ফ',
                     'দ্ধ', 'ন্থ', 'স্থ', 'ঞ্চ', 'ষ্ঠ', 'শ্চ', 'ল্ল', 'ল্ক', 'শ্ল', 'ত্ব', 'প্ট', 'ব্দ', 'ঙ্ক্ষ', 'স্ম',
                     'শ্ন', 'জ্জ', 'জ্ঞ', 'দ্গ', 'ল্ট', 'ম্ন', 'ম্ম', 'প্ল', 'ক্ট', 'ম্ভ', 'ব্ল', 'ন্ব', 'দ্ভ', 'ট্ট', 
                     'ক্ব', 'প্প', 'ক্ক', 'ষ্ক', 'গ্ন', 'হ্ম', 'ম্ল', 'প্ন', 'ব্ব', 'দ্ম', 'ক্ষ্ম', 'ষ্ণ', 'জ্ব', 'স্ল', 'শ্ম',
                     'ন্জ', 'ল্ফ', 'ঙ্ঘ', 'ব্জ', 'ঞ্ঝ', 'ন্ম', 'ক্ষ্ণ', 'ধ্ব', 'ফ্ট', 'স্প্ল', 'গ্ব', 'প্স', 'ষ্ম', 'গ্ম',
                     'ঙ্খ', 'ম্ফ', 'ব্ধ', 'হ্ন', 'ড্ড', 'ল্স', 'ত্থ', 'জ্জ্ব', 'হ্ণ', 'ন্ঠ', 'গ্ধ', 'হ্ব', 'ণ্ঠ', 'ণ্ব', 'দ্ধ্ব',
                     'ল্গ', 'ক্ন', 'ঘ্ন', 'গ্গ', 'হ্ল', 'ণ্ণ', 'ষ্ব', 'ন্দ্ব', 'দ্ঘ', 'ঞ্ছ', 'স্খ', 'ঙ্ম', 'ক্ম', 'ল্ব', 'ম্প্ল',
                     'থ্ব', 'শ্ত', 'ন্ত্ব', 'ণ্ঢ', 'শ্ছ', 'ষ্ফ', 'চ্ছ্ব', 'দ্দ্ব', 'স্ত্ব', 'ণ্ম', 'জ্ঝ', 'ধ্ন', 'ট্ব', 'চ্ঞ', 'ব্ভ']


    for pred, label in zip(predictions, labels):
        flag = True
        for conjunct in conjunct_list:
            if conjunct in label:
                flag = False

        if flag:
            simple_words += 1
            simple_predictions.append(pred)
            simple_labels.append(label)

    if verbose:
        print(f"Simple Words: {simple_words}")
        print()

    return simple_predictions, simple_labels


def print_model_size(model):
    torch.save(model.state_dict(), "temp.p")
    print('Model Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def run_benchmark(model, img_loader, device):
    elapsed = 0
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, _) in enumerate(img_loader):
        images = images.to(device)
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Inference time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed
