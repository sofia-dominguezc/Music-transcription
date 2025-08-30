from argparse import ArgumentParser
from training import train, test, MusicModel, LitMusicModel, save, load
from process_data import process_data, create_dataloader


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-a',
        '--action',
        type=str,
        choices=['train', 'train-val', 'test', 'process-train', 'process-test'],
        required=True,
        help='What the script will do',
    )
    # Training arguments
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate'
    )
    parser.add_argument(
        '--num-epochs', type=int, default=1, help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64, help='Batch size'
    )
    parser.add_argument(
        '--num-workers', type=int, default=0, help='Number of dataloader workers'
    )
    parser.add_argument(
        '--gamma', type=float, default=1, help='lr multiplier'
    )
    parser.add_argument(
        '--milestones', type=int, nargs='*', default=[], help='Scheduler milestones'
    )
    # Testing
    parser.add_argument(
        '--thresholds', type=float, nargs='*', default=[0.5],
        help='Thresholds of probability to use to calculate test accuracy. Must be in [0, 1]'
    )
    parser.add_argument(
        '--allowed-errors', type=int, nargs='*', default=[0],
        help='Allowed errors per frame to calculate test performance.'
    )
    parser.add_argument(
        '--test-dev', action='store_true', help='If true, runs test loop using dev model',
    )
    parser.add_argument(
        '--train-dev', action='store_true', help='If true, continues training dev model',
    )
    # NN architecture
    parser.add_argument(
        '--c', type=int, default=4, help='multiplier of channels for the conv layer'
    )
    parser.add_argument(
        '--n-heads', type=int, default=4, help='number of attention heads'
    )
    # Analysis arguments (editing means pre-processing data again)
    parser.add_argument(
        '--batch-seconds', type=float, default=4, help='Number of seconds in a batch of data',
    )
    parser.add_argument(
        '--bins-per-octave', type=int, default=48, help='Number of frequency bins per octave',
    )
    parser.add_argument(
        '--only-note-names', action='store_true', help='If true, processes notes modulo 12'
    )
    parser.add_argument(
        '--sr', type=int, default=22050, help='Sample rate'
    )
    parser.add_argument(
        '--hop-length', type=int, default=512, help='Hop length'
    )
    return parser.parse_args()


def save_model_decision(model):
    print("Do you want to save this model to model_weights.pth?")
    while True:
        action = input("Your response ('Y'/'N'): ")
        if action == 'Y':
            save(model)
            break
        elif action == 'N':
            break
        else:
            continue


if __name__ == "__main__":
    args = parse_args()

    if args.action in ["process-train", "process-test"]:
        split = args.action.split('-')[1]
        process_data(
            split, batch_seconds=args.batch_seconds, bins_per_octave=args.bins_per_octave,
            sr=args.sr, hop_length=args.hop_length, all_notes=not args.only_note_names,
        )
    elif args.action in ["train", "train-val"]:
        train_loader = create_dataloader("train", args.batch_size, args.num_workers)
        val_loader = (
            create_dataloader("test", args.batch_size, args.num_workers)
            if args.action.endswith('val') else None
        )
        model = MusicModel(
            c=args.c, n_freq=8*args.bins_per_octave, all_notes=not args.only_note_names,
            n_time=int(args.batch_seconds * args.sr / args.hop_length),
            n_heads=args.n_heads,
        )
        if args.train_dev:
            load(model, dev=True)
        train(
            model, train_loader=train_loader, val_loader=val_loader,
            lr=args.lr, total_epochs=args.num_epochs, pl_class=LitMusicModel,
            milestones=args.milestones, gamma=args.gamma,
        )
        save_model_decision(model)
    else:
        test_loader = create_dataloader("test", args.batch_size, args.num_workers)
        model = MusicModel(
            c=args.c, n_freq=8*args.bins_per_octave, all_notes=not args.only_note_names,
            n_time=int(args.batch_seconds * args.sr / args.hop_length),
            n_heads=args.n_heads,
        )
        load(model, dev=args.test_dev)
        test(
            model, test_loader, pl_class=LitMusicModel,
            thresholds=args.thresholds, allowed_errors=args.allowed_errors,
        )

# TODO: make the lazy dataloader so I can use more workers
