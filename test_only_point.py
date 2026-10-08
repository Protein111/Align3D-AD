import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import csv
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false)")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, act_layer=torch.nn.GELU):
        super().__init__()
        self.act_fcn = act_layer()
        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        self.projection = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)
        x = self.projection(x)
        x = self.act_fcn(x)
        x = self.output(x)
        return x

class FeatureAligner(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.global_mlp = FeatureProjectionMLP(in_features=feature_dim, out_features=feature_dim)
        self.local_mlp = FeatureProjectionMLP(in_features=feature_dim, out_features=feature_dim)

    def forward(self, global_feat, local_feat):
        aligned_global_feat = self.global_mlp(global_feat)
        aligned_local_feat = self.local_mlp(local_feat)
        return aligned_global_feat, aligned_local_feat

def back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index, ori_resolution = 336):
    # _, h, w, _ = d2_similarity_map.shape
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b*nv)[:,None], num_points).reshape(-1, ).cuda().long()
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits = d2_similarity_map[nbatch, :, yy, xx]
    point_logits = point_logits.reshape(b, nv, num_points, 2)
    vweights = torch.ones((1, nv, 1, 1))
    vweights = vweights.reshape(1, -1, 1, 1).to(point_logits.device)
    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    point_logits = point_logits * vweights * is_seen * non_zero_index
    mask = is_seen.bool() | (~non_zero_index.bool())
    point_logits = point_logits.sum(1)/ (mask.sum(1))
    point_logits = point_logits.reshape(b, h, w, 2)
    return point_logits

def test(args):
    features_list = args.features_list
    dataset_dir = args.data_path
    train_class = args.train_class

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("pretrained_weights/ViT-L-14-336px.pt", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform, target_transform_pc = get_transform(args)
    test_data = Dataset(root=dataset_dir, dataset_name = args.dataset, transform=preprocess, target_transform=target_transform, target_transform_pc=target_transform_pc, mode='test', is_all=True, point_size=args.point_size,)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    results = {}
    for obj in test_data.obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['integrate_pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['integrate_anomaly_maps'] = []

    ckpt_path = args.checkpoint_path
    ckpt_root, ckpt_ext = os.path.splitext(ckpt_path)
    if ckpt_path.endswith("_rgb.pth"):
        rgb_checkpoint_path = ckpt_path
        rendering_checkpoint_path = ckpt_path.replace("_rgb.pth", "_rendering.pth")
        checkpoint_name = os.path.basename(ckpt_root[:-4])
    elif ckpt_path.endswith("_rendering.pth"):
        rendering_checkpoint_path = ckpt_path
        rgb_checkpoint_path = ckpt_path.replace("_rendering.pth", "_rgb.pth")
        checkpoint_name = os.path.basename(ckpt_root[:-10])
    else:
        rgb_checkpoint_path = f"{ckpt_root}_rgb{ckpt_ext}"
        rendering_checkpoint_path = f"{ckpt_root}_rendering{ckpt_ext}"
        checkpoint_name = os.path.basename(ckpt_root)

    rgb_checkpoint = torch.load(rgb_checkpoint_path, map_location=device)
    rendering_checkpoint = torch.load(rendering_checkpoint_path, map_location=device)

    rgb_prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    if "rgb_prompt_learner" in rgb_checkpoint:
        rgb_prompt_learner.load_state_dict(rgb_checkpoint["rgb_prompt_learner"])
    elif "prompt_learner" in rgb_checkpoint:
        rgb_prompt_learner.load_state_dict(rgb_checkpoint["prompt_learner"])
    else:
        raise KeyError("No rgb prompt learner weights found in rgb checkpoint.")

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    if "prompt_learner" in rendering_checkpoint:
        prompt_learner.load_state_dict(rendering_checkpoint["prompt_learner"])
    else:
        raise KeyError("No rendering prompt learner weights found in rendering checkpoint.")

    aligner = FeatureAligner(model.visual.output_dim).to(device)
    if "aligner" in rgb_checkpoint:
        aligner.load_state_dict(rgb_checkpoint["aligner"])
    elif "aligner" in rendering_checkpoint:
        aligner.load_state_dict(rendering_checkpoint["aligner"])
    else:
        logger.warning("No aligner weights found in checkpoints; using randomly initialized aligner.")

    rgb_prompt_learner.to(device)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    aligner.eval()

    rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text = rgb_prompt_learner(cls_id=None)
    rgb_text_features = model.encode_text_learn(rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text).float()
    rgb_text_features = torch.stack(torch.chunk(rgb_text_features, dim=0, chunks=2), dim=1)
    rgb_text_features = rgb_text_features / rgb_text_features.norm(dim=-1, keepdim=True)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for items in tqdm(test_dataloader):
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        render_image = items['d2_render_img'].to(device)
        b, nv, c, h, w = render_image.shape
        render_image = render_image.reshape(-1, c, h, w)
        d2_3d_cor = items['d2_3d_cor'].to(device)
        non_zero_index = items['non_zero_index'].to(device)
        non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

        with torch.no_grad():
            image_features, patch_features = model.encode_image(render_image, features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            patch_feature = patch_features[0]
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

            aligned_global, aligned_local = aligner(image_features, patch_feature)
            aligned_global = aligned_global / aligned_global.norm(dim=-1, keepdim=True)
            aligned_local = aligned_local / aligned_local.norm(dim=-1, keepdim=True)

            rgb_text_probs = aligned_global.unsqueeze(1) @ rgb_text_features.permute(0, 2, 1)
            rgb_text_probs = (rgb_text_probs / 0.07).softmax(-1)
            rgb_text_probs = rgb_text_probs[:, 0, 1]
            rgb_text_probs = torch.chunk(rgb_text_probs, nv, dim=0)
            rgb_text_probs = torch.stack(rgb_text_probs, dim=1).mean(1)

            rendering_text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            rendering_text_probs = (rendering_text_probs / 0.07).softmax(-1)
            rendering_text_probs = rendering_text_probs[:, 0, 1]
            rendering_text_probs = torch.chunk(rendering_text_probs, nv, dim=0)
            rendering_text_probs = torch.stack(rendering_text_probs, dim=1).mean(1)

            rgb_similarity, _ = AnomalyCLIP_lib.compute_similarity(aligned_local, rgb_text_features[0])
            rgb_similarity_map = AnomalyCLIP_lib.get_similarity_map(rgb_similarity[:, 1:, :], args.image_size)

            rendering_similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            rendering_similarity_map = AnomalyCLIP_lib.get_similarity_map(rendering_similarity[:, 1:, :], args.image_size)

            d3_rgb_similarity_map = back_to_3d(rgb_similarity_map, d2_3d_cor, non_zero_index)
            d3_rendering_similarity_map = back_to_3d(rendering_similarity_map, d2_3d_cor, non_zero_index)

            d3_rgb_anomaly_map = d3_rgb_similarity_map[..., 1]
            rendering_anomaly_map = d3_rendering_similarity_map[..., 1]

            d3_rgb_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in d3_rgb_anomaly_map.detach().cpu()],dim=0,)
            rendering_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in rendering_anomaly_map.detach().cpu()],dim=0,)
            

            image_level_weight = args.image_level_weight
            integrate_anomaly_map = image_level_weight * d3_rgb_anomaly_map + (1-image_level_weight) * rendering_anomaly_map
            if args.need_blur:
                integrate_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in integrate_anomaly_map.detach().cpu()],dim=0,)
            

            integrate_text_probs = image_level_weight * rgb_text_probs + (1-image_level_weight) * rendering_text_probs

            results[cls_name[0]]['integrate_pr_sp'].extend((integrate_anomaly_map.max() + integrate_text_probs.detach().cpu()))

            point_level_weight = args.point_level_weight
            
            integrate_anomaly_map = point_level_weight * d3_rgb_anomaly_map + (1-point_level_weight) * rendering_anomaly_map
            if args.dataset == 'mvtec_pc_3d_rgb':
                integrate_anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in integrate_anomaly_map.detach().cpu()],dim=0,)
            results[cls_name[0]]['integrate_anomaly_maps'].append(integrate_anomaly_map)

    integrate_table_ls = []
    integrate_image_auroc_list = []
    integrate_image_ap_list = []
    integrate_pixel_auroc_list = []

    obj_list = [c for c in test_data.obj_list if c != train_class]
    for obj in obj_list:
        integrate_table = [obj]

        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['integrate_anomaly_maps'] = torch.cat(results[obj]['integrate_anomaly_maps']).detach().cpu().numpy()

        if args.metrics == 'image-level':
            integrate_image_auroc = image_level_metrics(results, obj, "image-auroc", modality='integrate_pr_sp')
            integrate_image_ap = image_level_metrics(results, obj, "image-ap", modality='integrate_pr_sp')
            integrate_table.append(str(np.round(integrate_image_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_ap * 100, decimals=1)))
            integrate_image_auroc_list.append(integrate_image_auroc)
            integrate_image_ap_list.append(integrate_image_ap)

        elif args.metrics == 'pixel-level':
            integrate_pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality='integrate_anomaly_maps')
            integrate_table.append(str(np.round(integrate_pixel_auroc * 100, decimals=1)))
            integrate_pixel_auroc_list.append(integrate_pixel_auroc)

        elif args.metrics == 'image-pixel-level':
            integrate_image_auroc = image_level_metrics(results, obj, "image-auroc", modality='integrate_pr_sp')
            integrate_image_ap = image_level_metrics(results, obj, "image-ap", modality='integrate_pr_sp')
            integrate_pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality='integrate_anomaly_maps')
            integrate_table.append(str(np.round(integrate_pixel_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_auroc * 100, decimals=1)))
            integrate_table.append(str(np.round(integrate_image_ap * 100, decimals=1)))
            integrate_image_auroc_list.append(integrate_image_auroc)
            integrate_image_ap_list.append(integrate_image_ap)
            integrate_pixel_auroc_list.append(integrate_pixel_auroc)

        integrate_table_ls.append(integrate_table)

    if args.metrics == 'image-level':
        metric_headers = ['objects', 'image_auroc', 'image_ap']

        integrate_table_ls.append(['mean',
            str(np.round(np.mean(integrate_image_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(integrate_image_ap_list) * 100, decimals=1)),
        ])
        integrate_results = tabulate(integrate_table_ls, headers=metric_headers, tablefmt="pipe")

    elif args.metrics == 'pixel-level':
        metric_headers = ['objects', 'pixel_auroc']

        integrate_table_ls.append(['mean',
            str(np.round(np.mean(integrate_pixel_auroc_list) * 100, decimals=1)),
        ])
        integrate_results = tabulate(integrate_table_ls, headers=metric_headers, tablefmt="pipe")

    elif args.metrics == 'image-pixel-level':
        metric_headers = ['objects', 'pixel_auroc', 'image_auroc', 'image_ap']

        integrate_table_ls.append(['mean',
            str(np.round(np.mean(integrate_pixel_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(integrate_image_auroc_list) * 100, decimals=1)),
            str(np.round(np.mean(integrate_image_ap_list) * 100, decimals=1)),
        ])
        integrate_results = tabulate(integrate_table_ls, headers=metric_headers, tablefmt="pipe")
    else:
        raise ValueError(f"Unsupported metrics mode: {args.metrics}")

    checkpoint_dir = os.path.dirname(rgb_checkpoint_path)
    result_txt_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_results.txt")
    full_report = (
        "===== Integrated Branch =====\n"
        f"{integrate_results}\n"
    )
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    # Save integrated branch metrics for aggregation/plotting.
    result_csv_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_results.csv")
    csv_rows = [["integrated"] + row for row in integrate_table_ls]

    with open(result_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["branch"] + metric_headers)
        writer.writerows(csv_rows)

    logger.info("\n%s", integrate_results)
    logger.info("Saved results table to %s", result_txt_path)
    logger.info("Saved results csv to %s", result_csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PointAD", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--train_class", type=str, default="cookie", help="zero shot or few shot")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    # hyperparameters
    parser.add_argument("--need_blur", type=str2bool, default=True, help="whether to apply gaussian blur to the anomaly maps")
    parser.add_argument("--image_level_weight", type=float, default=0.5, help="image-level anomaly score weight for integrated branch")
    parser.add_argument("--point_level_weight", type=float, default=0.5, help="point-level anomaly score weight for integrated branch")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
