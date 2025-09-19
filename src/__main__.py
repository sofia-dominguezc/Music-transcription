import argparse


def run_preprocess(args):
    from preprocess_data import process_data
    process_data(
        split=args.split,
        num_workers=args.num_workers,
        batch_seconds=args.batch_seconds,
        bins_per_note=args.bins_per_note,
        sr=args.sr,
        hop_length=args.hop_length,
        all_notes=~args.only_note_names,
        n_batches=args.n_batches,
    )


def run_train(args):
    from training import train, test, MusicTransformer, load
    from dataloaders import create_lazy_dataloader

    model = MusicTransformer(
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        c=args.c,
        embed_dim=args.embed_dim,
    )

    if args.load_weights == "main":
        load(model, dev=False)
    elif args.load_weights == "dev":
        load(model, dev=True)

    train_loader = create_lazy_dataloader(
        split="train", batch_size=args.batch_size, num_workers=args.num_workers
    )
    val_loader = create_lazy_dataloader(
        split="test", batch_size=1, num_workers=0
    )

    train(
        model,
        train_loader,
        lr=args.lr,
        total_epochs=args.epochs,
        val_loader=val_loader,
    )

    test(
        model,
        val_loader,
        allowed_errors=args.allowed_errors,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Music transcription project"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Preprocessing ---
    process_parser = subparsers.add_parser("preprocess", help="Preprocess data")
    process_parser.add_argument("--split", type=str, default="test")
    process_parser.add_argument("--num-workers", type=int, default=8)
    process_parser.add_argument("--batch-seconds", type=float, default=1)
    process_parser.add_argument("--bins-per-note", type=int, default=4)
    process_parser.add_argument("--sr", type=int, default=22050)
    process_parser.add_argument("--hop-length", type=int, default=512)
    process_parser.add_argument("--only-note-names", action="store_true")
    process_parser.add_argument("--n-batches", type=int, default=60)

    # --- Training ---
    train_parser = subparsers.add_parser("train", help="Train and test model")
    train_parser.add_argument("--n-layers", type=int, default=4)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--head-dim", type=int, default=32)
    train_parser.add_argument("--c", type=int, default=3)
    train_parser.add_argument("--embed-dim", type=int, default=192)
    train_parser.add_argument("--load-weights", options=["main", "dev"], default=None)

    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=5e-4)
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--allowed-errors", type=int, nargs="+", default=[0, 1, 2])

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "train":
        run_train(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
