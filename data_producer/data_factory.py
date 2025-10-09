from data_producer.data_loader import Dataset_MTS
from torch.utils.data import DataLoader


def create_mts_loader(args):
    if args.status == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.train_batch_size
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.test_batch_size

    data_set = Dataset_MTS(
        data_name=args.data_name,
        data_path=args.data_path,
        cont_len=args.cont_len,
        pred_len=args.pred_len,
        status=args.status,
        task=args.task,
        target=args.target,
        freq=args.freq)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    print(f"Sample number in {args.status} set: {len(data_set)}")
    return data_loader