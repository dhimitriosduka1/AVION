import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch

import decord


def datetime2sec(str):
    hh, mm, ss = str.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(
        np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode="valid"
    )
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()


def get_video_reader(
    videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params
):
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0],
            height=rrc_params[0],
            use_rrc=True,
            scale_min=rrc_params[1][0],
            scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0],
            height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader


def video_loader(
    root,
    vid,
    ext,
    second,
    end_second,
    chunk_len=300,
    fps=30,
    clip_length=32,
    threads=1,
    fast_rrc=False,
    rrc_params=(224, (0.5, 1.0)),
    fast_rcc=False,
    rcc_params=(224,),
    jitter=False,
):
    assert fps > 0, "fps should be greater than 0"

    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, "{}.{}".format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc,
            rrc_params=rrc_params,
            fast_rcc=fast_rcc,
            rcc_params=rcc_params,
        )
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(
            frame_offset,
            min(frame_offset + total_duration, len(vr)),
            num_segments=clip_length,
            jitter=jitter,
        )

        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()

        return torch.from_numpy(frames.astype(np.float32))

    else:
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        while True:
            video_filename = osp.join(
                root, "{}.{}".format(vid, ext), "{}.{}".format(chunk_end, ext)
            )
            if not osp.exists(video_filename):
                print("{} does not exists!".format(video_filename))
                chunk_end -= chunk_len
            else:
                vr = decord.VideoReader(video_filename)
                end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
                assert chunk_start <= chunk_end
                break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length,
            jitter=jitter,
        )
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(
                filter(
                    lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps),
                    frame_ids,
                )
            )
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, "{}.{}".format(vid, ext), "{}.{}".format(chunk, ext)),
                num_threads=threads,
                fast_rrc=fast_rrc,
                rrc_params=rrc_params,
                fast_rcc=fast_rcc,
                rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(
            root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids
        )
        return res


def video_loader_by_frames(
    root, vid, frame_ids, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params
):
    vr = get_video_reader(
        videoname=osp.join(root, vid),
        num_threads=num_threads,
        fast_rrc=fast_rrc,
        rrc_params=rrc_params,
        fast_rcc=fast_rcc,
        rcc_params=rcc_params,
    )
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed

        if self.dataset == "ego4d":
            with open(metadata, "rb") as f:
                self.samples = pickle.load(f)
        elif self.dataset in ["ek100_cls", "ek100_mir"]:
            video_list = glob.glob(osp.join(self.root, "*/*.MP4"))
            fps_dict = {
                video: decord.VideoReader(video + "/0.MP4").get_avg_fps()
                for video in video_list
            }
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(
                        row[5]
                    )
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    vid_path = "{}/{}".format(pid, vid)
                    fps = fps_dict[osp.join(self.root, vid_path + ".MP4")]
                    # start_frame = int(np.round(fps * start_timestamp))
                    # end_frame = int(np.ceil(fps * end_timestamp))
                    self.samples.append(
                        (
                            vid_path,
                            start_timestamp,
                            end_timestamp,
                            fps,
                            narration,
                            verb,
                            noun,
                        )
                    )
            if self.dataset == "ek100_mir":
                self.metadata_sentence = pd.read_csv(
                    metadata[: metadata.index(".csv")] + "_sentence.csv"
                )
                if "train" in metadata:
                    self.relevancy_mat = pickle.load(
                        open(
                            osp.join(
                                osp.dirname(metadata),
                                "relevancy",
                                "caption_relevancy_EPIC_100_retrieval_train.pkl",
                            ),
                            "rb",
                        )
                    )
                elif "test" in metadata:
                    self.relevancy_mat = pickle.load(
                        open(
                            osp.join(
                                osp.dirname(metadata),
                                "relevancy",
                                "caption_relevancy_EPIC_100_retrieval_test.pkl",
                            ),
                            "rb",
                        )
                    )
                else:
                    raise ValueError(
                        '{} should contain either "train" or "test"!'.format(metadata)
                    )
                self.relevancy = 0.1
        elif self.dataset == "egtea":
            len_dict_path = osp.join(osp.dirname(metadata), "video_len_dict.pkl")
            if not osp.exists(len_dict_path):
                print("Generating video length dict...")
                video_list = glob.glob(osp.join(self.root, "*/*"))
                len_dict = {
                    video: len(decord.VideoReader(video)) for video in video_list
                }

                with open(len_dict_path, "wb") as f:
                    pickle.dump(len_dict, f)

                print(f"Saved video length dict to {len_dict_path}")
            else:
                with open(len_dict_path, "rb") as f:
                    len_dict = pickle.load(f)

                print(f"Loaded video length dict from {len_dict_path}")

            vn_list, labels = [], []
            for row in open(osp.join(osp.dirname(metadata), "action_idx.txt")):
                row = row.strip()
                vn = int(row.split(" ")[-1])
                vn_list.append(vn)
                narration = " ".join(row.split(" ")[:-1])
                labels.append(narration.replace("_", " ").lower())

            mapping_act2narration = {
                vn: narration for vn, narration in zip(vn_list, labels)
            }

            self.samples = []
            with open(metadata) as f:
                for row in f:
                    clip_id, action_idx = row.strip().split(" ")[:2]
                    video_id = "-".join(clip_id.split("-")[:3])
                    vid_relpath = osp.join(video_id, "{}.mp4".format(clip_id))
                    vid_fullpath = osp.join(
                        self.root, video_id, "{}.mp4".format(clip_id)
                    )
                    self.samples.append(
                        (
                            vid_relpath,
                            0,
                            len_dict[vid_fullpath],
                            mapping_act2narration[int(action_idx)],
                        )
                    )
        elif self.dataset == "charades_ego":
            video_list = glob.glob(osp.join(self.root, "*.mp4"))

            fps_dict_path = osp.join(osp.dirname(metadata), "fps_dict.pkl")
            if osp.exists(fps_dict_path):
                with open(fps_dict_path, "rb") as f:
                    fps_dict = pickle.load(f)
                print(f"Loaded fps dict from {fps_dict_path}")
            else:
                print("Generating fps dict...")
                fps_dict = {
                    video: decord.VideoReader(video).get_avg_fps()
                    for video in video_list
                }
                with open(fps_dict_path, "wb") as f:
                    pickle.dump(fps_dict, f)
                print(f"Saved fps dict to {fps_dict_path}")

            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)
                for row in csv_reader:
                    video_id = row[0]
                    if self.is_trimmed:
                        for action_tuple in row[9].split(";"):
                            if not action_tuple:
                                continue
                            action, start_timestamp, end_timestamp = action_tuple.split(
                                " "
                            )
                            start_timestamp, end_timestamp = float(
                                start_timestamp
                            ), float(end_timestamp)
                            vid_path = "{}.mp4".format(video_id)
                            fps = fps_dict[osp.join(self.root, vid_path)]
                            start_frame = int(np.round(fps * start_timestamp))
                            end_frame = int(np.ceil(fps * end_timestamp))
                            self.samples.append(
                                (vid_path, start_frame, end_frame, action)
                            )
                    else:
                        if not row[9]:
                            action_list = []
                        else:
                            action_list = [
                                action_tuple.split(" ")[0]
                                for action_tuple in row[9].split(";")
                            ]
                        vid_path = "{}.mp4".format(video_id)
                        fps = fps_dict[osp.join(self.root, vid_path)]
                        duration = fps * float(row[10])
                        self.samples.append((vid_path, 0, duration, action_list))
        else:
            raise NotImplementedError

    def get_raw_item(
        self,
        i,
        is_training=True,
        num_clips=1,
        chunk_len=300,
        clip_length=32,
        clip_stride=2,
        sparse_sample=False,
        narration_selection="random",
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
    ):
        if self.dataset == "ego4d":
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames = video_loader(
                self.root,
                vid,
                "mp4",
                start_second,
                end_second,
                chunk_len=chunk_len,
                clip_length=clip_length,
                threads=threads,
                fast_rrc=fast_rrc,
                rrc_params=rrc_params,
                fast_rcc=fast_rcc,
                rcc_params=rcc_params,
                jitter=is_training,
            )
            if isinstance(narration, list):
                if narration_selection == "random":
                    narration = random.choice(narration)
                elif narration_selection == "concat":
                    narration = ". ".join(narration)
                elif narration_selection == "list":
                    pass
                else:
                    raise ValueError
            return frames, narration
        elif self.dataset == "ek100_mir":
            vid_path, start_second, end_second, fps, narration, verb, noun = (
                self.samples[i]
            )
            frames = video_loader(
                self.root,
                vid_path,
                "MP4",
                start_second,
                end_second,
                chunk_len=chunk_len,
                fps=fps,
                clip_length=clip_length,
                threads=threads,
                fast_rrc=fast_rrc,
                rrc_params=rrc_params,
                fast_rcc=fast_rcc,
                rcc_params=rcc_params,
                jitter=is_training,
            )
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[
                    0
                ].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if (
                        pos < len(self.metadata_sentence)
                        and pos < self.relevancy_mat.shape[1]
                    ):
                        return frames, (
                            self.metadata_sentence.iloc[pos][1],
                            self.relevancy_mat[i][pos],
                        )
            else:
                return frames, (narration, 1)
        elif self.dataset == "ek100_cls":
            vid_path, start_second, end_second, fps, narration, verb, noun = (
                self.samples[i]
            )
            frames = video_loader(
                self.root,
                vid_path,
                "MP4",
                start_second,
                end_second,
                chunk_len=chunk_len,
                fps=fps,
                clip_length=clip_length,
                threads=threads,
                fast_rrc=fast_rrc,
                rrc_params=rrc_params,
                fast_rcc=fast_rcc,
                rcc_params=rcc_params,
                jitter=is_training,
            )
            return frames, "{}:{}".format(verb, noun)
        elif self.dataset == "egtea":
            vid_path, start_frame, end_frame, sentence = self.samples[i]
            if is_training:
                assert num_clips == 1
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        list(np.arange(0, end_frame)),
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )
                    zeros = torch.zeros(
                        (clip_length * clip_stride - end_frame, *frames.shape[1:])
                    )
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                else:
                    start_id = np.random.randint(
                        0, end_frame - clip_length * clip_stride + 1
                    )
                    frame_ids = np.arange(
                        start_id, start_id + clip_length * clip_stride, clip_stride
                    )
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        frame_ids,
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        list(np.arange(0, end_frame)),
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )
                    zeros = torch.zeros(
                        (clip_length * clip_stride - end_frame, *frames.shape[1:])
                    )
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(
                        0, end_frame - clip_length * clip_stride, num_clips, dtype=int
                    ):
                        frame_ids.extend(
                            np.arange(
                                start_id,
                                start_id + clip_length * clip_stride,
                                clip_stride,
                            )
                        )
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        frame_ids,
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )

            return frames, sentence
        elif self.dataset == "charades_ego":
            vid_path, start_frame, end_frame, action_list = self.samples[i]
            if sparse_sample:
                frame_ids = get_frame_ids(
                    start_frame,
                    end_frame,
                    num_segments=num_clips * clip_length,
                    jitter=is_training,
                )
                frames = video_loader_by_frames(
                    self.root,
                    vid_path,
                    frame_ids,
                    threads,
                    fast_rrc,
                    rrc_params,
                    fast_rcc,
                    rcc_params,
                )
            else:
                if end_frame < clip_length * clip_stride:
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        list(np.arange(0, end_frame)),
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )
                    zeros = torch.zeros(
                        (clip_length * clip_stride - end_frame, *frames.shape[1:])
                    )
                    frames = torch.cat((frames, zeros), dim=0)
                    frames = frames[::clip_stride]
                    frames = frames.repeat(num_clips, 1, 1, 1)
                else:
                    frame_ids = []
                    for start_id in np.linspace(
                        0, end_frame - clip_length * clip_stride, num_clips, dtype=int
                    ):
                        frame_ids.extend(
                            np.arange(
                                start_id,
                                start_id + clip_length * clip_stride,
                                clip_stride,
                            )
                        )
                    frames = video_loader_by_frames(
                        self.root,
                        vid_path,
                        frame_ids,
                        threads,
                        fast_rrc,
                        rrc_params,
                        fast_rcc,
                        rcc_params,
                    )
            return frames, action_list
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(
        self,
        dataset,
        root,
        metadata,
        transform=None,
        is_training=True,
        tokenizer=None,
        chunk_len=300,
        clip_length=32,
        clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        subsample_stride=None,
    ):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        frames, caption = self.get_raw_item(
            i,
            is_training=self.is_training,
            chunk_len=self.chunk_len,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
        )

        # ek100_mir will also output relevancy value
        if isinstance(caption, tuple):
            caption, relevancy = caption
        else:
            relevancy = 0.0

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)[0]

        if isinstance(caption, tuple):
            caption, mask = caption
            return frames, caption, mask, relevancy
        else:
            return frames, caption, relevancy


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self,
        dataset,
        root,
        metadata,
        transform=None,
        is_training=True,
        label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32,
        clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True,
    ):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
            i,
            is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.0
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label


def get_downstream_dataset(
    transform, crop_size, args, subset="train", label_mapping=None
):
    if subset == "train":
        return VideoClassyDataset(
            args.dataset,
            args.root,
            args.train_metadata,
            transform,
            is_training=True,
            label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=args.fused_decode_crop,
            rrc_params=(crop_size, (0.5, 1.0)),
        )
    elif subset == "val":
        return VideoClassyDataset(
            args.dataset,
            args.root,
            args.val_metadata,
            transform,
            is_training=False,
            label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rcc=args.fused_decode_crop,
            rcc_params=(crop_size,),
            is_trimmed=not args.dataset == "charades_ego",
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")
