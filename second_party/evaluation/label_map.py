import csv
import os


def generate_label_map(root, dataset):
    if dataset == "charades_ego":
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open(os.path.join(root, "Charades_v1_classes.txt")) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == "egtea":
        print("=> preprocessing egtea action label space")
        labels = []
        with open(os.path.join(root, "action_idx.txt")) as f:
            for row in f:
                row = row.strip()
                narration = " ".join(row.split(" ")[:-1])
                labels.append(narration.replace("_", " ").lower())
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act
