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
    parser.add_argument('--dataset', type=str, default='synthetic', choices=["mjsynth", "synthetic", "banglawriting", "bnhtrd", "apurba"], help='dataset name')
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


    apurba_data_domain = "cc" # cc | lp | tw
    # Computer Composed | letterpress | typewriter
    # apurba_train_csv = f"january3_csvs/{apurba_data_domain}_jan3_train.csv"
    apurba_train_csv = "january3_csvs/cc_with_synth_punct_num.csv"
    apurba_valid_csv = f"january3_csvs/{apurba_data_domain}_jan3_valid.csv"
    # Computer Composed Combined Words and Characters
    # apurba_train_csv = "january3_csvs/cc_comb_jan3_train.csv"
    # apurba_valid_csv = "january3_csvs/cc_comb_jan3_valid.csv"
    # All Printed
    # apurba_train_csv = "csvs/printed_august_train.csv"
    # apurba_valid_csv = "csvs/printed_august_valid.csv"
    # Handwriting Isolated Words
    # apurba_train_csv = "csvs/hwiw_september_train.csv"
    # apurba_valid_csv = "csvs/hwiw_september_valid.csv"
    # Handwriting Running Words
    # apurba_train_csv = "csvs/hwrw_september_train.csv"
    # apurba_valid_csv = "csvs/hwrw_september_valid.csv"
    # Computer Composed merged with synthetic words, numbers and punctuations
    # apurba_train_csv = "csvs/cc_august_train_with_synthocr_punct_num.csv"
    # apurba_valid_csv = "csvs/cc_august_valid.csv"

    parser.add_argument('--apurba_train_csv', type=str, default=apurba_train_csv, help='apurba train data labels csv path')
    parser.add_argument('--apurba_valid_csv', type=str, default=apurba_valid_csv, help='apurba validation data labels csv path')

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

    parser.add_argument('--dataset', type=str, default='test', choices=["test", "synthetic", "banglawriting", "bnhtrd", "apurba"], help='test dataset name')
    parser.add_argument('--epoch', type=int, default=29, help='epoch of the trained model')
    parser.add_argument('--grapheme_rep', type=str, default='ads', choices=["ads", "vds", "naive"], help='grapheme representation method')
    parser.add_argument('--grapheme_dict_path', type=str, default='./ckpts/crnnpaper/ads/inv_grapheme_dict_synth.pickle', help='Load grapheme dict of the trained model')
    parser.add_argument('--model_dir', type=str, default='./ckpts/crnnpaper/ads', help='trained model weights saved directory')

    # apurba csv path
    apurba_data_domain = "cc" # cc | lp | tw
    dataset_segment = "test" # all | test

    # Computer Composed | letterpress | typewriter
    apurba_csv_path = f"january3_csvs/{apurba_data_domain}_jan3_{dataset_segment}.csv"

    # apurba_csv_path = "january3_csvs/ccc_jan3_all.csv" # Computer composed character all
    # apurba_csv_path = "january3_csvs/ccc_jan3_test.csv" # Computer composed character test
    # apurba_csv_path = "january3_csvs/cc_comb_jan3_test.csv" # Computer composed words and characters test
    # apurba_csv_path = "august_snap/csvs/hwiw_september_all.csv" # Handwriting Isolated All
    # apurba_csv_path = "august_snap/csvs/hwiw_september_test.csv" # Handwriting Isolated Test

    # BN-HTRd CSV
    bnhtrd_csv = "BN-HTRd_all.csv"
    # bnhtrd_csv = "BN-HTRd_test.csv"

    # BanglaWriting labels file
    banglawriting_txt = "raw_labels.txt"
    # banglawriting_txt = "raw_labels_test.txt"
    # banglawriting_txt = "converted_labels.txt"
    # banglawriting_txt = "converted_labels_test.txt"

    parser.add_argument('--test_protocol', type=str, default=3, help='test protocol')
    parser.add_argument('--apurba_csv_path', type=str, default=apurba_csv_path, help='apurba data labels csv path')
    parser.add_argument('--bnhtrd_csv', type=str, default=bnhtrd_csv, help='BN-HTRd labels csv file')
    parser.add_argument('--banglawriting_txt', type=str, default=banglawriting_txt, help='BanglaWriting labels text file')

    parser.add_argument('--vis_res', type=bool, default=True, help='save the predictions and corresponding ground truths in a text file')
    parser.add_argument('--visres_fn', type=str, default='results_test_ads.csv', help='training vis_res file name')

    return parser
