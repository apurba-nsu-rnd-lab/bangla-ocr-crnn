from argparse import ArgumentParser


def get_train_config():
    """This function includes training options."""
    parser = ArgumentParser()

    # Transfer Learning
    parser.add_argument('--transfer_learning', type=bool, default=False, help='perform transfer learning using pretrained weights')
    parser.add_argument('--pretrn_model_path', type=str, default='./ckpts/mixed/epoch_29_best_val.pth', help='trained model weights saved path')

    # Distillation
    parser.add_argument('--teacher_path', type=str, default='./ckpts/cc500k_ft262Q/epoch_0_best_val.pth', help='trained teacher weights saved path')
    parser.add_argument('--temperature', type=float, default=3.0, help='Distillation Temperature')
    parser.add_argument('--alpha', type=float, default=0.02, help='Distillation Alpha')
    parser.add_argument('--rectify', type=bool, default=False, help='Rectify teacher predictions')

    # Resume Training
    parser.add_argument('--continue_from_last', type=bool, default=False, help='Continue training from a checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts/epoch_9.tar', help='trained model weights saved path')

    parser.add_argument('--qat', type=bool, default=False, help='perform quantization aware training')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=["mjsynth", "synthetic", "banglawriting", "bnhtrd"], help='dataset name')
    parser.add_argument('--augment', type=bool, default=True, help='apply data augmentation')
    parser.add_argument('--grapheme_rep', type=str, default='vds', choices=["ads", "vds", "naive"], help='grapheme representation method')
    parser.add_argument('--load_dict', type=bool, default=True, help='Load grapheme dict instead of creating anew') # Must set to True if transfer_learning is True

    parser.add_argument('--epochs', type=bool, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size') # 256 for synthetic, 128 for bnhtrd and banglawriting
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to fetch data')
    parser.add_argument('--optimizer', type=str, default='adam', choices=["adam", "adadelta", "sgd"], help='choose the optimizer')
    # During Quantization Aware Training, the LR should be 1-10% of the LR used for training without quantization
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate') # 5e-5 for quantized model, 1e-3 for bnhtrd and banglawriting
    parser.add_argument('--scheduler', type=bool, default=False, help='Use learning rate scheduler')
    # CosineAnnealingWarmRestarts LR Scheduler params
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate') # 1e-6 for quantized model, 5e-4 for bnhtrd and banglawriting
    parser.add_argument('--t0', type=int, default=15, help='number of iterations for the first restart')

    parser.add_argument('--vis_res', type=bool, default=False, help='save the predictions and corresponding ground truths in a text file')
    parser.add_argument('--train_visres_fn', type=str, default='results_training.csv', help='training vis_res file name')
    parser.add_argument('--valid_visres_fn', type=str, default='results_validation.csv', help='validation vis_res file name')
    parser.add_argument('--train_metrics_fn', type=str, default='metrics_training.csv', help='training metrics file name')
    parser.add_argument('--valid_metrics_fn', type=str, default='metrics_validation.csv', help='validation metrics file name')

    return parser


def get_test_config():
    """This function includes testing options."""
    parser = ArgumentParser()

    parser.add_argument('--qat', type=bool, default=False, help='load quantized model')
    parser.add_argument('--device', type=str, default='auto', choices=["auto", "cpu"], help='select auto to choose cuda if avaulable or test on cpu only')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    parser.add_argument('--dataset', type=str, default='test', choices=["test", "synthetic", "banglawriting", "bnhtrd"], help='test dataset name')
    parser.add_argument('--epoch', type=int, default=29, help='epoch of the trained model')
    parser.add_argument('--grapheme_rep', type=str, default='ads', choices=["ads", "vds", "naive"], help='grapheme representation method')
    parser.add_argument('--grapheme_dict_path', type=str, default='./ckpts/crnnpaper/ads/inv_grapheme_dict_synth.pickle', help='Load grapheme dict of the trained model')
    parser.add_argument('--model_dir', type=str, default='./ckpts/crnnpaper/ads', help='trained model weights saved directory')

    # BN-HTRd CSV
    bnhtrd_csv = "BN-HTRd_all.csv"
    # bnhtrd_csv = "BN-HTRd_test.csv"

    # BanglaWriting labels file
    banglawriting_txt = "raw_labels.txt"
    # banglawriting_txt = "raw_labels_test.txt"
    # banglawriting_txt = "converted_labels.txt"
    # banglawriting_txt = "converted_labels_test.txt"

    parser.add_argument('--test_protocol', type=str, default=3, help='test protocol')
    parser.add_argument('--bnhtrd_csv', type=str, default=bnhtrd_csv, help='BN-HTRd labels csv file')
    parser.add_argument('--banglawriting_txt', type=str, default=banglawriting_txt, help='BanglaWriting labels text file')

    parser.add_argument('--vis_res', type=bool, default=True, help='save the predictions and corresponding ground truths in a text file')
    parser.add_argument('--visres_fn', type=str, default='results_test_ads.csv', help='training vis_res file name')

    return parser
