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
        raise RuntimeError("Could not read RecordIO header at index 0: %s" % rec_path)

    header0, _ = mx.recordio.unpack(packed)
    return record, header0


def label_to_list(label):
    if isinstance(label, np.ndarray):
        return label.tolist()
    if isinstance(label, (list, tuple)):
        return list(label)
    return [label]


def record_count_from_header(header0):
    labels = label_to_list(header0.label)
    if len(labels) < 1:
        raise RuntimeError("Invalid RecordIO header: no label[0]")
    return int(labels[0]) - 1


def copy_if_exists(src, dst, overwrite=False):
    if not os.path.exists(src):
        print("WARNING: not found, skipping: %s" % src)
        return

    if os.path.isdir(src):
        if os.path.exists(dst):
            if not overwrite:
                raise FileExistsError(
                    "Destination already exists: %s\nUse --overwrite to replace it." % dst
                )
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and not overwrite:
            raise FileExistsError(
                "Destination already exists: %s\nUse --overwrite to replace it." % dst
            )
        shutil.copy2(src, dst)


def build_top_quality_mask(score_csv, percentage):
    df = pd.read_csv(score_csv)

    required = {"identity", "crfiqa_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "CRFIQA CSV is missing required column(s): %s. Found: %s"
            % (sorted(missing), list(df.columns))
        )

    df = df[["identity", "crfiqa_score"]].copy()

    if df["identity"].isna().any() or df["crfiqa_score"].isna().any():
        raise ValueError("CRFIQA CSV contains NaN values.")

    df["identity"] = df["identity"].astype(np.int64)
    df["crfiqa_score"] = df["crfiqa_score"].astype(np.float64)
    df = df.sort_values("identity", kind="mergesort").reset_index(drop=True)

    n = len(df)
    expected = np.arange(n, dtype=np.int64)
    actual = df["identity"].to_numpy(dtype=np.int64)

    if not np.array_equal(actual, expected):
        raise ValueError(
            "The 'identity' column must be a unique 0-based global image index "
            "0..N-1. It is not contiguous in this CSV."
        )

    if not (0.0 < percentage <= 100.0):
        raise ValueError("--percentage must be > 0 and <= 100.")

    keep_n = max(1, int(round(n * percentage / 100.0)))
    scores = df["crfiqa_score"].to_numpy()
    ranked = np.argsort(-scores, kind="mergesort")
    selected = ranked[:keep_n]

    mask = np.zeros(n, dtype=bool)
    mask[selected] = True

    cutoff = float(scores[selected].min())
    print("CRFIQA rows       : %d" % n)
    print("Keep percentage   : %.4f%%" % percentage)
    print("Images to keep    : %d" % keep_n)
    print("Images to remove  : %d" % (n - keep_n))
    print("Lowest kept score : %.8f" % cutoff)

    return mask, scores


def filter_gallery(input_root, output_root, num_ID, keep_mask, overwrite=False):
    src_dir = os.path.join(input_root, "local_gallery_%d" % num_ID)
    dst_dir = os.path.join(output_root, "local_gallery_%d" % num_ID)

    if os.path.exists(dst_dir):
        if not overwrite:
            raise FileExistsError(
                "Destination already exists: %s\nUse --overwrite to replace it." % dst_dir
            )
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    src, header0 = read_record_header(src_dir, "test.rec", "test.idx")
    n_gallery = record_count_from_header(header0)

    if n_gallery != len(keep_mask):
        src.close()
        raise ValueError(
            "CRFIQA row count (%d) != local gallery image count (%d). "
            "The CSV and dataset do not align." % (len(keep_mask), n_gallery)
        )

    h0_labels = label_to_list(header0.label)
    gallery_num_ID = int(h0_labels[1]) if len(h0_labels) > 1 else int(num_ID)

    dst = mx.recordio.MXIndexedRecordIO(
        os.path.join(dst_dir, "test.idx"),
        os.path.join(dst_dir, "test.rec"),
        "w",
    )

    new_idx = 1
    for src_idx in tqdm(
        range(1, n_gallery + 1), ncols=120, desc="Filtering gallery"
    ):
        global_idx = src_idx - 1
        if not keep_mask[global_idx]:
            continue

        packed = src.read_idx(src_idx)
        header, img = mx.recordio.unpack(packed)
        label = int(header.label)
        new_header = mx.recordio.IRHeader(0, label, new_idx, 0)
        dst.write_idx(new_idx, mx.recordio.pack(new_header, img))
        new_idx += 1

    new_header0 = mx.recordio.IRHeader(2, [new_idx, gallery_num_ID], 0, 0)
    dst.write_idx(0, mx.recordio.pack(new_header0, bytes(0)))

    src.close()
    dst.close()
    print("Gallery kept      : %d / %d" % (new_idx - 1, n_gallery))


def get_client_image_counts(input_root, num_ID, num_client):
    train_dir = os.path.join(
        input_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )
    counts = []
    header_meta = []

    for i in range(num_client):
        client_dir = os.path.join(train_dir, "client_%04d" % i)
        src, header0 = read_record_header(client_dir, "train.rec", "train.idx")
        count = record_count_from_header(header0)
        labels = label_to_list(header0.label)

        client_num_ID = int(labels[1]) if len(labels) > 1 else num_ID // num_client
        start_ID = int(labels[2]) if len(labels) > 2 else i * client_num_ID

        counts.append(count)
        header_meta.append((client_num_ID, start_ID))
        src.close()

    return counts, header_meta


def filter_clients(
    input_root, output_root, num_ID, num_client, keep_mask, scores, overwrite=False
):
    src_train_dir = os.path.join(
        input_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )
    dst_train_dir = os.path.join(
        output_root, "split_train_i%04dc%04d" % (num_ID, num_client)
    )

    counts, header_meta = get_client_image_counts(input_root, num_ID, num_client)
    total_train = int(sum(counts))

    if total_train != len(keep_mask):
        raise ValueError(
            "CRFIQA row count (%d) != total client-training image count (%d). "
            "The CSV must be ordered exactly as client_0000, client_0001, ... "
            "and image index 1..N inside each client."
            % (len(keep_mask), total_train)
        )

    if os.path.exists(dst_train_dir):
        if not overwrite:
            raise FileExistsError(
                "Destination already exists: %s\nUse --overwrite to replace it."
                % dst_train_dir
            )
        shutil.rmtree(dst_train_dir)
    os.makedirs(dst_train_dir, exist_ok=True)

    global_offset = 0
    total_kept = 0
    kept_scores = []

    for i in range(num_client):
        src_client_dir = os.path.join(src_train_dir, "client_%04d" % i)
        dst_client_dir = os.path.join(dst_train_dir, "client_%04d" % i)
        os.makedirs(dst_client_dir, exist_ok=True)

        src, _ = read_record_header(src_client_dir, "train.rec", "train.idx")
        dst = mx.recordio.MXIndexedRecordIO(
            os.path.join(dst_client_dir, "train.idx"),
            os.path.join(dst_client_dir, "train.rec"),
            "w",
        )

        n_client = counts[i]
        client_num_ID, start_ID = header_meta[i]
        new_idx = 1

        for src_idx in tqdm(
            range(1, n_client + 1),
            ncols=120,
            desc="Client %04d" % i,
            leave=True,
        ):
            global_idx = global_offset + (src_idx - 1)
            if not keep_mask[global_idx]:
                continue

            packed = src.read_idx(src_idx)
            header, img = mx.recordio.unpack(packed)
            local_label = int(header.label)
            new_header = mx.recordio.IRHeader(0, local_label, new_idx, 0)
            dst.write_idx(new_idx, mx.recordio.pack(new_header, img))
            kept_scores.append(scores[global_idx])
            new_idx += 1

        new_header0 = mx.recordio.IRHeader(
            2, [new_idx, client_num_ID, start_ID], 0, 0
        )
        dst.write_idx(0, mx.recordio.pack(new_header0, bytes(0)))

        kept_client = new_idx - 1
        total_kept += kept_client
        global_offset += n_client

        src.close()
        dst.close()

        print(
            "Client %04d kept: %d / %d (%.2f%%)"
            % (i, kept_client, n_client, 100.0 * kept_client / max(1, n_client))
        )

    pd.DataFrame(
        {
            "identity": np.arange(total_kept, dtype=np.int64),
            "crfiqa_score": np.asarray(kept_scores, dtype=np.float64),
        }
    ).to_csv(os.path.join(dst_train_dir, "CRFIQA_local.csv"), index=False)

    print("Training kept     : %d / %d" % (total_kept, total_train))
    print("Saved CRFIQA CSV  : %s" % os.path.join(dst_train_dir, "CRFIQA_local.csv"))


def main():
    parser = argparse.ArgumentParser(
        description="Create a FedFR split using the highest CRFIQA-scoring images."
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

    keep_mask, scores = build_top_quality_mask(score_csv, args.percentage)

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
        input_root, output_root, args.num_ID, keep_mask, overwrite=args.overwrite
    )
    filter_clients(
        input_root,
        output_root,
        args.num_ID,
        args.num_client,
        keep_mask,
        scores,
        overwrite=args.overwrite,
    )

    print("\nDone.")
    print("FedFR-compatible output written to: %s" % output_root)


if __name__ == "__main__":
    main()
