

def modify_args(args):

    args.input_size = (1088, 4000)
    args.test_size = (1088, 4000)
    args.num_classes = 2
    args.eval_interval = 1
    args.print_interval = 20
    args.data_dir = '/home/jerry/data/Micro_A/A_loushi/10-23-loushi/labelme/labelme_gj_update_bbaug_coco'
    args.train_ann = "instances_train2017.json"
    args.val_ann = "instances_val2017.json"
    args.max_epoch = 250
    args.no_aug_epochs = 15

    return args
