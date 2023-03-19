import json
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from configs import get_train_config
from data.dataset_utils import (decode_prediction, encode_banglawriting_data,
                                encode_bnhtrd, encode_mjsynth_data,
                                encode_synth_data, get_padded_labels)
from data.datasets import (BanglaWritingDataset, BNHTRDataset,
                           MJSynthDataset, SynthDataset)
from models.model_vgg import get_crnn
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from utils_eval import count_parameters, recognition_metrics

# from sklearn.metrics import classification_report


#seeding for reproducability
random_seed = 33
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

args = get_train_config().parse_args()
print(args)
with open('./results/training_config.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.load_dict:
# load grapheme dictionary of pre-trained model
    with open(f"./grapheme_dicts/inv_grapheme_dict_synth_{args.grapheme_rep}.pickle", 'rb') as handle:
        inv_grapheme_dict = pickle.load(handle)

if args.dataset == "mjsynth":

    # Path to dataset
    DATA_PATH = "/home/blank/projects/kdwr/datasets/mnt/ramdisk/max/90kDICT32px"
    # DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/synthetic_hw"

    # Preprocess the data and get grapheme dictionary and labels
    _, img_paths_tr, words_tr, labels_tr, lengths_tr = encode_mjsynth_data(DATA_PATH, "annotation_train_clean.txt", inv_grapheme_dict)
    _, img_paths_val, words_val, labels_val, lengths_val = encode_mjsynth_data(DATA_PATH, "annotation_val_clean.txt", inv_grapheme_dict)

    train_dataset = MJSynthDataset(img_paths_tr, transform=args.augment)
    valid_dataset = MJSynthDataset(img_paths_val, transform=args.augment)

elif args.dataset == "synthetic":

    # Path to dataset
    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/synthetic_words"
    # DATA_PATH_2 = "/home/ec2-user/word_level_ocr/pritom/datasets/synthetic_wordlist_punct_num_pruned"
    # DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/synthetic_hw"

    # Preprocess the data and get grapheme dictionary and labels
    _, words_tr, labels_tr, lengths_tr = encode_synth_data(os.path.join(DATA_PATH, "train_labels.txt"), representation=args.grapheme_rep)
    # _, words_tr, labels_tr, lengths_tr = encode_synth_data(os.path.join(DATA_PATH_2, "train_labels.txt"), inv_grapheme_dict, representation=args.grapheme_rep)
    _, words_val, labels_val, lengths_val = encode_synth_data(os.path.join(DATA_PATH, "valid_labels.txt"), inv_grapheme_dict, representation=args.grapheme_rep)

    train_dataset = SynthDataset(os.path.join(DATA_PATH, "train"), transform=args.augment)
    # train_dataset = SynthDataset(os.path.join(DATA_PATH_2, "train"), transform=args.augment)
    valid_dataset = SynthDataset(os.path.join(DATA_PATH, "valid"), transform=args.augment)

elif args.dataset == "banglawriting":

    # Path to dataset
    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/BanglaWriting_words/"
    # TRAIN_TXT = "converted_labels_train.txt"
    # VALID_TXT = "converted_labels_valid.txt"
    TRAIN_TXT = "raw_labels_train.txt"
    VALID_TXT = "raw_labels_valid.txt"

    # Preprocess the data and get grapheme dictionary and labels
    _, words_tr, labels_tr, lengths_tr = encode_banglawriting_data(os.path.join(DATA_PATH, TRAIN_TXT), inv_grapheme_dict, representation=args.grapheme_rep)
    _, words_val, labels_val, lengths_val = encode_banglawriting_data(os.path.join(DATA_PATH, VALID_TXT), inv_grapheme_dict, representation=args.grapheme_rep)

    # train_dataset = BanglaWritingDataset(os.path.join(DATA_PATH, "converted"), os.path.join(DATA_PATH, TRAIN_TXT), transform=args.augment)
    # valid_dataset = BanglaWritingDataset(os.path.join(DATA_PATH, "converted"), os.path.join(DATA_PATH, VALID_TXT), transform=args.augment)
    train_dataset = BanglaWritingDataset(os.path.join(DATA_PATH, "raw"), os.path.join(DATA_PATH, TRAIN_TXT), transform=args.augment)
    valid_dataset = BanglaWritingDataset(os.path.join(DATA_PATH, "raw"), os.path.join(DATA_PATH, VALID_TXT), transform=args.augment)

elif args.dataset == "bnhtrd":
	
    # Path to dataset
    DATA_PATH = "/home/ec2-user/word_level_ocr/pritom/datasets/handwriting/BN-HTRd/"
    TRAIN_CSV = "BN-HTRd_train.csv"
    VAL_CSV = "BN-HTRd_valid.csv"

    # Preprocess the data and get grapheme dictionary and labels
    _, words_tr, labels_tr, lengths_tr = encode_bnhtrd(os.path.join(DATA_PATH, TRAIN_CSV), inv_grapheme_dict, representation=args.grapheme_rep)
    _, words_val, labels_val, lengths_val = encode_bnhtrd(os.path.join(DATA_PATH, VAL_CSV), inv_grapheme_dict, representation=args.grapheme_rep)
	
    train_dataset = BNHTRDataset(DATA_PATH, TRAIN_CSV, transform=args.augment)
    valid_dataset = BNHTRDataset(DATA_PATH, VAL_CSV, transform=args.augment)


# Save grapheme dictionary
# with open("./ckpts/inv_grapheme_dict_synth.pickle", 'wb') as handle:
#     pickle.dump(inv_grapheme_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(inv_grapheme_dict)
#print(len(inv_grapheme_dict))

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
validation_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# Define model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = get_crnn(len(inv_grapheme_dict)+1, qatconfig=('fbgemm' if args.qat else None))
model = model.to(device)

if args.transfer_learning:
    model.load_state_dict(torch.load(args.pretrn_model_path, map_location=device))

# print(model)
count_parameters(model)

print("Backbone summary: ")
summary(model.backbone, (1, 32, 128), batch_size=-1)

criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
criterion = criterion.to(device)

def train():

    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=0.9, dampening=0, weight_decay=1e-05)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=1e-5)

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    # new_lr = lr * factor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=1, threshold=0.01, min_lr=args.min_lr, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 25], gamma=0.5, verbose=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=1, eta_min=args.min_lr, verbose=True)

    if args.transfer_learning:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        for name, param in model.rnn.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = True
    
    best_valid_epoch = 999
    best_valid_loss = float("inf")

    for epoch in range(args.epochs):

        model.train()

        if args.transfer_learning:
            if epoch > args.epochs // 3:
                for name, param in model.rnn.named_parameters():
                    param.requires_grad = True
            if epoch > args.epochs * 2 // 3 :
                for name, param in model.backbone.named_parameters():
                    param.requires_grad = True

        #necessary variables
        # y_true = []
        # y_pred = []
        decoded_preds = []
        ground_truths = []

        print(f"***Epoch: {epoch}/{args.epochs-1}***")
        batch_loss = 0

        for inp, idx in tqdm(train_loader):

            inp = inp.to(device)
            # inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            # print(idxs)
            words, labels, labels_size = get_padded_labels(idxs, words_tr, labels_tr, lengths_tr)

            preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)
            loss = criterion(preds, labels, preds_size, labels_size)
            batch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        train_loss = batch_loss/args.batch_size
        print("Epoch Training loss: ", train_loss)
        print()

        rec_results = recognition_metrics(decoded_preds, ground_truths, vis_res=args.vis_res, file_name=args.train_visres_fn)

        ##################### Generate Training Report ##############################
        # report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1),
        #                                zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])

        # with pd.ExcelWriter("./results/classification_report_training.xlsx", engine='openpyxl', mode=('w' if epoch==0 else 'a')) as writer:
        #     pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{epoch}'

        metrics = pd.DataFrame([{'epoch': epoch,
                                 'crr': rec_results['crr'],
                                 'wrr': rec_results['wrr'],
                                 'total_ned': rec_results['total_ned'],
                                 'abs_match': rec_results['abs_match'],
                                 'train_loss': train_loss
                                 }])

        metrics.to_csv(f"./results/{args.train_metrics_fn}",
                       mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False))

        #############################################################################

        valid_loss = validate(epoch)

        scheduler.step(valid_loss)

        ################################### SAVE MODELS #####################################
        # save best validation model
        if (valid_loss < best_valid_loss):
            if os.path.exists(f"./ckpts/epoch_{best_valid_epoch}_best_val.pth"):
                os.remove(f"./ckpts/epoch_{best_valid_epoch}_best_val.pth")

            best_valid_epoch = epoch
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"./ckpts/epoch_{best_valid_epoch}_best_val.pth")

        # save models between epoch (##-##)
        # if epoch >= 25 and epoch <= 30:
        #     torch.save(model.state_dict(), f"./ckpts/epoch{epoch}.pth")



def validate(epoch):

    # evaluate model:
    model.eval()

    with torch.no_grad():
        # y_true = []
        # y_pred = []
        decoded_preds = []
        ground_truths = []

        print(f"***Epoch: {epoch}***")
        batch_loss = 0

        for inp, idx in tqdm(validation_loader):

            inp = inp.to(device)
            # inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            words, labels, labels_size = get_padded_labels(idxs, words_val, labels_val, lengths_val)

            preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
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

        valid_loss = batch_loss/args.batch_size
        print("Epoch Validation loss: ", valid_loss) #batch_size denominator 32
        print()

        rec_results = recognition_metrics(decoded_preds, ground_truths, vis_res=args.vis_res, file_name=args.valid_visres_fn)

        ##################### Generate Validation Report ##############################
        
        # report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
        #                                zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])

        # with pd.ExcelWriter("./results/classification_report_validation.xlsx", engine='openpyxl', mode=('w' if epoch==0 else 'a')) as writer:  
        #     pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{epoch}')

        metrics = pd.DataFrame([{'epoch': epoch,
                                 'crr': rec_results['crr'],
                                 'wrr': rec_results['wrr'],
                                 'total_ned': rec_results['total_ned'],
                                 'abs_match': rec_results['abs_match'],
                                 'valid_loss': valid_loss
                                 }])
        
        metrics.to_csv(f"./results/{args.valid_metrics_fn}",
                       mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False))

        ###############################################################################

        return valid_loss



if __name__ == "__main__":
    # execute only if run as a script
    train()
