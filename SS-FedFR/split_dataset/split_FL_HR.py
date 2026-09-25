import os
import shutil
import argparse

import mxnet as mx
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_record_header(rec_dir, rec_name, idx_name):
    idx_path = os.path.join(rec_dir, idx_name)
    rec_path = os.path.join(rec_dir, rec_name)

    if not os.path.isfile(idx_path):
        raise FileNotFoundError("Missing RecordIO index file: %s" % idx_path)
    if not os.path.isfile(rec_path):
        raise FileNotFoundError("Missing RecordIO record file: %s" % rec_path)

    record = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
    packed = record.read_idx(0)
    if packed is None:
        record.close()
        raise RuntimeError("Could not read RecordIO header: %s" % rec_path)

    header, _ = mx.recordio.unpack(packed)
    return record, header


def label_list(label):
    if isinstance(label, np.ndarray):
        return label.tolist()
    if isinstance(label, (list, tuple)):
        return list(label)
    return [label]


def record_count(header):
    labels = label_list(header.label)
    if not labels:
        raise RuntimeError("Invalid RecordIO header.")
    return int(labels[0]) - 1


def copy_if_exists(src, dst, overwrite=False):
    if not os.path.exists(src):
        print("WARNING: not found, skipping: %s" % src)
        return

    if os.path.isdir(src):
        if os.path.exists(dst):
            if not overwrite:
                raise FileExistsError(
                    "Destination exists: %s\nUse --overwrite to replace it." % dst
                )
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(
                "Destination exists: %s\nUse --overwrite to replace it." % dst
            )
        shutil.copy2(src, dst)


def build_low_headpose_mask(score_csv, percentage):
    df = pd.read_csv(score_csv)

    if "sum_deg" not in df.columns:
        raise ValueError(
            "Headpose CSV must contain 'sum_deg'. Found: %s" % list(df.columns)
        )
    if df["sum_deg"].isna().any():
        raise ValueError("Headpose CSV contains NaN values in 'sum_deg'.")
    if not 0 < percentage <= 100:
        raise ValueError("--percentage must be > 0 and <= 100.")

    scores = df["sum_deg"].to_numpy(dtype=np.float64)
    n = len(df)
    keep_n = max(1, int(round(n * percentage / 100.0)))

    selected = np.argsort(scores, kind="mergesort")[:keep_n]
    mask = np.zeros(n, dtype=bool)
    mask[selected] = True

    print("Headpose rows      : %d" % n)
    print("Keep percentage   : %.2f%%" % percentage)
    print("Images to keep    : %d" % keep_n)
    print("Images to remove  : %d" % (n - keep_n))
    print("Highest kept score: %.8f deg" % scores[selected].max())

    return mask, df


def filter_gallery(input_root, output_root, num_ID, mask, overwrite=False):
    src_dir = os.path.join(input_root, "local_gallery_%d" % num_ID)
    dst_dir = os.path.join(output_root, "local_gallery_%d" % num_ID)

    if os.path.exists(dst_dir):
        if not overwrite:
            raise FileExistsError(
                "Destination exists: %s\nUse --overwrite to replace it." % dst_dir
            )
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    src, header0 = read_record_header(src_dir, "test.rec", "test.idx")
    n = record_count(header0)

    if n != len(mask):
        src.close()
        raise ValueError(
            "Headpose row count (%d) != gallery image count (%d). "
            "The CSV must follow the same global image order." % (len(mask), n)
        )

    labels = label_list(header0.label)
    gallery_num_ID = int(labels[1]) if len(labels) > 1 else num_ID

    dst = mx.recordio.MXIndexedRecordIO(
        os.path.join(dst_dir, "test.idx"),
        os.path.join(dst_dir, "test.rec"),
        "w",
    )

    new_idx = 1
    for src_idx in tqdm(range(1, n + 1), ncols=120, desc="Filtering gallery"):
        global_idx = src_idx - 1
        if not mask[global_idx]:
            continue

        packed = src.read_idx(src_idx)
        header, img = mx.recordio.unpack(packed)
        dst.write_idx(
            new_idx,
            mx.recordio.pack(
                mx.recordio.IRHeader(0, int(header.label), new_idx, 0), img
            ),
        )
        new_idx += 1

    dst.write_idx(
        0,
        mx.recordio.pack(
            mx.recordio.IRHeader(2, [new_idx, gallery_num_ID], 0, 0), bytes(0)
        ),
    )

    src.close()
    dst.close()
    print("Gallery kept      : %d / %d" % (new_idx - 1, n))


def get_client_info(input_root, num_ID, num_client):
    train_dir = os.path.join(
        input_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )

    counts, meta = [], []
    for i in range(num_client):
        client_dir = os.path.join(train_dir, "client_%04d" % i)
        src, header = read_record_header(client_dir, "train.rec", "train.idx")
        labels = label_list(header.label)

        counts.append(record_count(header))
        client_num_ID = int(labels[1]) if len(labels) > 1 else num_ID // num_client
        start_ID = int(labels[2]) if len(labels) > 2 else i * client_num_ID
        meta.append((client_num_ID, start_ID))
        src.close()

    return counts, meta


def filter_clients(
    input_root, output_root, num_ID, num_client, mask, headpose_df, overwrite=False
):
    src_train = os.path.join(
        input_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )
    dst_train = os.path.join(
        output_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )

    counts, meta = get_client_info(input_root, num_ID, num_client)
    total = int(sum(counts))

    if total != len(mask):
        raise ValueError(
            "Headpose row count (%d) != total client image count (%d). "
            "Rows must be ordered as client_0000, client_0001, ... and "
            "RecordIO index 1..N within each client." % (len(mask), total)
        )

    if os.path.exists(dst_train):
        if not overwrite:
            raise FileExistsError(
                "Destination exists: %s\nUse --overwrite to replace it." % dst_train
            )
        shutil.rmtree(dst_train)
    os.makedirs(dst_train, exist_ok=True)

    offset = 0
    total_kept = 0
    kept_rows = []

    for i in range(num_client):
        src_dir = os.path.join(src_train, "client_%04d" % i)
        dst_dir = os.path.join(dst_train, "client_%04d" % i)
        os.makedirs(dst_dir, exist_ok=True)

        src, _ = read_record_header(src_dir, "train.rec", "train.idx")
        dst = mx.recordio.MXIndexedRecordIO(
            os.path.join(dst_dir, "train.idx"),
            os.path.join(dst_dir, "train.rec"),
            "w",
        )

        n = counts[i]
        client_num_ID, start_ID = meta[i]
        new_idx = 1

        for src_idx in tqdm(
            range(1, n + 1), ncols=120, desc="Client %04d" % i, leave=True
        ):
            global_idx = offset + src_idx - 1
            if not mask[global_idx]:
                continue

            packed = src.read_idx(src_idx)
            header, img = mx.recordio.unpack(packed)
            dst.write_idx(
                new_idx,
                mx.recordio.pack(
                    mx.recordio.IRHeader(0, int(header.label), new_idx, 0), img
                ),
            )
            kept_rows.append(global_idx)
            new_idx += 1

        dst.write_idx(
            0,
            mx.recordio.pack(
                mx.recordio.IRHeader(
                    2, [new_idx, client_num_ID, start_ID], 0, 0
                ),
                bytes(0),
            ),
        )

        kept = new_idx - 1
        total_kept += kept
        offset += n

        src.close()
        dst.close()

        print(
            "Client %04d kept: %d / %d (%.2f%%)"
            % (i, kept, n, 100.0 * kept / max(1, n))
        )

    filtered_df = headpose_df.iloc[kept_rows].reset_index(drop=True)
    csv_out = os.path.join(dst_train, "headpose.csv")
    filtered_df.to_csv(csv_out, index=False)

    print("Training kept     : %d / %d" % (total_kept, total))
    print("Saved headpose CSV: %s" % csv_out)


def main():
    parser = argparse.ArgumentParser(
        description="Create a FedFR split using the lowest headpose sum_deg scores."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--score_csv", required=True)
    parser.add_argument("--num_client", type=int, default=40)
    parser.add_argument("--num_ID", type=int, default=4000)
    parser.add_argument("--percentage", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)
    score_csv = os.path.abspath(args.score_csv)

    if input_root == output_root:
        raise ValueError("--output_dir must be different from --input_dir.")

    os.makedirs(output_root, exist_ok=True)

    print("Input  : %s" % input_root)
    print("Output : %s" % output_root)
    print("Scores : %s\n" % score_csv)

    mask, headpose_df = build_low_headpose_mask(score_csv, args.percentage)

    copy_if_exists(
        os.path.join(input_root, "local_veri_%d" % args.num_ID),
        os.path.join(output_root, "local_veri_%d" % args.num_ID),
        overwrite=args.overwrite,
    )
    copy_if_exists(
        os.path.join(input_root, "ID2idx.pickle"),
        os.path.join(output_root, "ID2idx.pickle"),
        overwrite=args.overwrite,
    )

    filter_gallery(
        input_root, output_root, args.num_ID, mask, overwrite=args.overwrite
    )
    filter_clients(
        input_root,
        output_root,
        args.num_ID,
        args.num_client,
        mask,
        headpose_df,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print("FedFR-compatible output written to: %s" % output_root)


if __name__ == "__main__":
    main()
