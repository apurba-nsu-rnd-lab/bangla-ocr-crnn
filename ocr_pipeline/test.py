import json
import os
import pickle
import random

import numpy as np
import torch
from configs import get_test_config
from data.dataset_utils import (decode_prediction, encode_banglawriting_data,
                                encode_bnhtrd, encode_synth_data, get_padded_labels)
from data.datasets import (BanglaWritingDataset, BNHTRDataset,
                           SynthDataset)
from models.model_vgg import get_crnn
from torch import quantization
# import pandas as pd
from torchsummary import summary
from tqdm import tqdm
from utils_eval import (count_parameters, get_simple_words,
                        mispredicted_conjunct_count, print_model_size,
                        recognition_metrics, recognition_metrics_atlength,
                        run_benchmark)

# from sklearn.metrics import classification_report


# Reproducability
random_seed = 33
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


#Commandline arguments
args = get_test_config().parse_args()
print(args)
# with open('./results/test_config.json', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)


with open(args.grapheme_dict_path, 'rb') as handle:
    inv_grapheme_dict = pickle.load(handle)

print(inv_grapheme_dict)


if args.dataset == "test":

    # Printed Dataset
    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/test_protocols"
    _, words_te, labels_te, lengths_te = encode_synth_data(os.path.join(DATA_PATH, "labels/protocol{}_labels.txt".format(args.test_protocol)),
                                                           inv_grapheme_dict, representation=args.grapheme_rep)
    test_dataset = SynthDataset(os.path.join(DATA_PATH, "images/protocol{}".format(args.test_protocol)))

elif args.dataset == "synthetic":

    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/num_punct_test"
    _, words_te, labels_te, lengths_te = encode_synth_data(os.path.join(DATA_PATH, "test_labels.txt"), inv_grapheme_dict, representation=args.grapheme_rep)
    test_dataset = SynthDataset(os.path.join(DATA_PATH, "test"))

elif args.dataset == "banglawriting":

    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/BanglaWriting_words/"
    TEST_TXT = args.banglawriting_txt
    _, words_te, labels_te, lengths_te = encode_banglawriting_data(os.path.join(DATA_PATH, TEST_TXT), inv_grapheme_dict, representation=args.grapheme_rep)
    test_dataset = BanglaWritingDataset(os.path.join(DATA_PATH, "raw"), os.path.join(DATA_PATH, TEST_TXT)) # converted or raw

elif args.dataset == "bnhtrd":

    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/BN-HTRd/"
    CSV_PATH = args.bnhtrd_csv
    _, words_te, labels_te, lengths_te = encode_bnhtrd(os.path.join(DATA_PATH, CSV_PATH), inv_grapheme_dict, representation=args.grapheme_rep)
    test_dataset = BNHTRDataset(DATA_PATH, CSV_PATH)


inference_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

###########################################################################################

####################For graph later(set number of samples)########################

# num_samples = len(test_dataset)

# file_namelist = []

# for i in range(num_samples):
#     #print(ocr_dataset[i][1])
#     file_namelist.append(test_dataset[i][1])
###################################################################################

#########################Model Import and parameter print##################################

if args.device == 'auto':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
elif args.device == 'cpu':
    device = torch.device('cpu')

model = get_crnn(len(inv_grapheme_dict)+1, qatconfig=('fbgemm' if args.qat else None))
model = model.to(device)
# print(model)

#Path to model
model.load_state_dict(torch.load(os.path.join(args.model_dir, 'epoch_{}_best_val.pth'.format(args.epoch)), map_location=device))

# loss function
criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
criterion = criterion.cuda()

def test(model):

    model.eval()

    with torch.no_grad():
        # y_true = []
        # y_pred = []
        decoded_preds = []
        ground_truths = []

        total_wer = 0

        print("***Epoch: {}***".format(args.epoch))
        batch_loss = 0
        for inp, idx in tqdm(inference_loader):
            inp = inp.to(device)
            # inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            words , labels, labels_size = get_padded_labels(idxs, words_te, labels_te, lengths_te)            
            #print(labels)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)

            preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)

            #validation loss
            loss = criterion(preds, labels, preds_size, labels_size)
            #print(loss)
            batch_loss += loss.item()
            #print(loss.item())

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()

            # labels = labels.detach().numpy()
            # for pred, label in zip(preds, labels):
                # decoded_pred, _ = decode_prediction(pred, inv_grapheme_dict)
                # for x, y in zip(decoded_pred, label):
                #     y_pred.append(x)
                #     y_true.append(y)

            for pred, gt in zip(preds, words):
                _, decoded_string = decode_prediction(pred, inv_grapheme_dict)

                decoded_preds.append(decoded_string)
                ground_truths.append(gt)


        valid_loss = batch_loss/batch_size
        print("Epoch Validation loss: ", valid_loss)
        print()

        rec_results = recognition_metrics(decoded_preds, ground_truths, vis_res=args.vis_res, file_name=args.visres_fn)
        # recognition_metrics_atlength(decoded_preds, ground_truths)
        mispredicted_conjunct_count(decoded_preds, ground_truths)
        simple_preds, simple_labels = get_simple_words(decoded_preds, ground_truths)
        recognition_metrics(simple_preds, simple_labels, vis_res=args.vis_res, file_name=args.visres_fn)

        print("End of Epoch {}".format(args.epoch))
        print("\n")

        # report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1),
        #                                zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
        # pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel("./results/classification_report_test.xlsx")
        


if __name__ == "__main__":
    # execute only if run as a script

    count_parameters(model)
    # print("Backbone summary: ")
    # summary(model.backbone, (1, 32, 128), batch_size=-1)

    if args.device == 'cpu' and args.qat:
        # Quantized model (QAT and DQ applied)
        quantized_model = quantization.convert(model.eval(), inplace=False)
        quantized_model.rnn = quantization.quantize_dynamic(quantized_model.rnn, {torch.nn.LSTM}, dtype=torch.qint8)
        quantized_model.embedding = quantization.quantize_dynamic(quantized_model.embedding, {torch.nn.Linear}, dtype=torch.qint8)

        print_model_size(quantized_model)
        run_benchmark(quantized_model, inference_loader, device)
        test(quantized_model)

    else:
        print_model_size(model)
        run_benchmark(model, inference_loader, device)
        test(model)
        