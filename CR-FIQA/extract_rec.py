# ----------------------------------------------------------------------------------------------------------------------
# GLOBAL EXTRACT
# ----------------------------------------------------------------------------------------------------------------------
# import os
# import cv2
# import numpy as np
# import mxnet as mx
# from tqdm import tqdm
#
# # Change this for other dataset
# path = "/home/master/FedFR-main/ms1m_split/split_pretrain_6000/client_0000/train.rec"
# idx_path = "/home/master/FedFR-main/ms1m_split/split_pretrain_6000/client_0000/train.idx"
# image_size = (112, 112)
# outpath = "/home/master/FedFR-main/ms1m_split/split_pretrain_6000/client_0000"
#
# # Read the .rec file using MXNet
# imgrec = mx.recordio.MXIndexedRecordIO(idx_path, path, 'r')
#
# dataset_name = path.split("/")[-1].split(".")[0]
# rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
# outpath = os.path.join(outpath, dataset_name)
# if not os.path.exists(outpath):
#     os.makedirs(outpath)
#     os.makedirs(os.path.join(outpath, "images"))
#
# print("extract:", dataset_name)
#
# # Read the .rec file metadata
# header, _ = mx.recordio.unpack(imgrec.read_idx(0))
# imgidx = list(range(1, int(header.label[0])))
#
# # Initialize issame_list based on some logic or external data source
# # For demonstration purposes, we'll assume a dummy issame_list where each pair is not the same
# # Make sure it has the correct length (half the total number of images)
# issame_list = [0] * (len(imgidx) // 2)
#
# pair_list = np.zeros((len(issame_list), 3), np.int16)
#
# txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")
#
# for idx in tqdm(imgidx):
#     _bin = imgrec.read_idx(idx)
#     header, s = mx.recordio.unpack(_bin)
#     img = mx.image.imdecode(s)
#
#     # Ensure image is converted to RGB
#     if img is not None and img.size > 0:
#         img = cv2.cvtColor(img.asnumpy(), cv2.COLOR_BGR2RGB)
#
#         if img.shape[1] != image_size[0]:
#             img = mx.image.resize_short(img, image_size[0])
#
#         cv2.imwrite(os.path.join(outpath, "images", "%05d.jpg" % (idx-1)), img)
#         txt_file.write(os.path.join(rel_img_path, "%05d.jpg" % (idx-1)) + "\n")
#
#         if (idx-1) % 2 == 0 and ((idx-1)//2) < len(issame_list):
#             pair_list[(idx-1)//2] = [idx-1, idx, issame_list[(idx-1)//2]]
#     else:
#         print(f"Warning: Image at index {idx} could not be loaded.")
#
# txt_file.close()
# np.savetxt(os.path.join(outpath, "pair_list.txt"), pair_list, fmt='%05d')
# print("pair_list saved")

# ----------------------------------------------------------------------------------------------------------------------
# LOCAL EXTRACT
# ----------------------------------------------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import mxnet as mx
from tqdm import tqdm

def process_client(client_id):
    # Change this for other dataset
    base_path = "/home/master/FedFR-main/ms1m_split/split_train_i4000c0040"
    client_folder = f"client_{client_id:04d}"
    path = os.path.join(base_path, client_folder, "train.rec")
    idx_path = os.path.join(base_path, client_folder, "train.idx")
    image_size = (112, 112)
    outpath = os.path.join(base_path, client_folder)

    # Read the .rec file using MXNet
    imgrec = mx.recordio.MXIndexedRecordIO(idx_path, path, 'r')

    dataset_name = path.split("/")[-1].split(".")[0]
    rel_img_path = os.path.join(outpath.split("/")[-1], dataset_name, "images")
    outpath = os.path.join(outpath, dataset_name)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        os.makedirs(os.path.join(outpath, "images"))

    print(f"extracting: {dataset_name} for client: {client_id:04d}")

    # Read the .rec file metadata
    header, _ = mx.recordio.unpack(imgrec.read_idx(0))
    imgidx = list(range(1, int(header.label[0])))

    # Initialize issame_list based on some logic or external data source
    # For demonstration purposes, we'll assume a dummy issame_list where each pair is not the same
    # Make sure it has the correct length (half the total number of images)
    issame_list = [0] * (len(imgidx) // 2)

    pair_list = np.zeros((len(issame_list), 3), np.int16)

    txt_file = open(os.path.join(outpath, "image_path_list.txt"), "w")

    for idx in tqdm(imgidx):
        _bin = imgrec.read_idx(idx)
        header, s = mx.recordio.unpack(_bin)
        img = mx.image.imdecode(s)

        # Ensure image is converted to RGB
        if img is not None and img.size > 0:
            img = cv2.cvtColor(img.asnumpy(), cv2.COLOR_BGR2RGB)

            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])

            cv2.imwrite(os.path.join(outpath, "images", "%05d.jpg" % (idx-1)), img)
            txt_file.write(os.path.join(rel_img_path, "%05d.jpg" % (idx-1)) + "\n")

            if (idx-1) % 2 == 0 and ((idx-1)//2) < len(issame_list):
                pair_list[(idx-1)//2] = [idx-1, idx, issame_list[(idx-1)//2]]
        else:
            print(f"Warning: Image at index {idx} could not be loaded.")

    txt_file.close()
    np.savetxt(os.path.join(outpath, "pair_list.txt"), pair_list, fmt='%05d')
    print("pair_list saved")

# Process clients from client_0000 to client_0039
for client_id in range(40):
    process_client(client_id)
