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
import numpy as np
import os
import random
from utils import get_transform


def _contrastive_two_logits(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    """Compute -log exp(cos(a,p)) / (exp(cos(a,p)) + exp(cos(a,n))) averaged over batch.

    Shapes:
        anchor/positive/negative: (..., dim) where leading dims match.
    """
    sim_pos = F.cosine_similarity(anchor, positive, dim=-1)
    sim_neg = F.cosine_similarity(anchor, negative, dim=-1)
    logits = torch.stack([sim_pos, sim_neg], dim=-1).reshape(-1, 2)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features=None, out_features=None, act_layer=torch.nn.GELU):
        # CFM Anomaly Detection via CLIP Feature Alignment
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
        # local_feat shape: (b*nv, num_patches, feature_dim)
        aligned_global_feat = self.global_mlp(global_feat)
        aligned_local_feat = self.local_mlp(local_feat)
        return aligned_global_feat, aligned_local_feat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train(args):
    logger = get_logger(args.save_path)
    preprocess, target_transform, target_transform_pc = get_transform(args)
    # device = args.device if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    model, _ = AnomalyCLIP_lib.load("pretrained_weights/ViT-L-14-336px.pt", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, target_transform_pc = target_transform_pc, dataset_name = args.dataset, train_dataset_name = args.train_dataset_name, point_size = args.point_size)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  ##########################################################################################
    rgb_prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    rgb_prompt_learner.to(device)
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    # Learn an explicit rendering->RGB feature mapping for both global and local tokens.
    aligner = FeatureAligner(model.visual.output_dim).to(device)
    ##########################################################################################
    optimizer_align = torch.optim.Adam(
        list(aligner.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_rgb_total = torch.optim.Adam(
        list(rgb_prompt_learner.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )
    optimizer_total = torch.optim.Adam(
        list(prompt_learner.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    model.eval()

    # Stage 1: optimize align_loss only (no prompt-loss optimization).
    rgb_prompt_learner.eval()
    prompt_learner.eval()
    aligner.train()

    # Cache CLIP features once to avoid repeated expensive extraction in stage 1.
    stage1_cache = []
    with torch.no_grad():
        for items in tqdm(train_dataloader, desc="build_stage1_cache", leave=False):
            render_image = items['d2_render_img'].to(device) # shape: (b, nv, c, h, w)
            rgb_render_image = items['d2_rgb_render_img'].to(device) # shape: (b, nv, c, h, w)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w) # shape: (b*nv, c, h, w)
            rgb_render_image = rgb_render_image.reshape(-1, c, h, w) # shape: (b*nv, c, h, w)

            image_features, patch_features = model.encode_image(render_image, args.features_list, DPAM_layer = 20)
            rgb_render_image_features, rgb_render_patch_features = model.encode_image(rgb_render_image, args.features_list, DPAM_layer = 20)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            rgb_render_image_features = rgb_render_image_features / rgb_render_image_features.norm(dim=-1, keepdim=True)

            patch_feature = patch_features[0]
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            rgb_patch_feature = rgb_render_patch_features[0]
            rgb_patch_feature = rgb_patch_feature / rgb_patch_feature.norm(dim=-1, keepdim=True)

            # Build patch-to-patch similarity matrix S for each view (render vs pseudo-RGB)
            # then reduce it to per-render-patch similarity s by averaging over RGB patches.
            # Shapes:
            #   patch_feature/rgb_patch_feature: (b*nv, num_patches, dim)
            #   s_matrix: (b*nv, num_patches, num_patches)
            #   s: (b*nv, num_patches)
            s_matrix = patch_feature @ rgb_patch_feature.transpose(1, 2)
            s = s_matrix.mean(dim=-1)

            stage1_cache.append((
                image_features.detach().cpu(),
                patch_feature.detach().cpu(),
                rgb_render_image_features.detach().cpu(),
                rgb_patch_feature.detach().cpu(),
                s.detach().cpu(),
            ))

    # In stage 1, enable weighted local alignment only for the last `epoch_1_local_loss` epochs.
    stage1_local_weight_start_epoch = max(0, args.epoch_1 - args.epoch_1_local_loss)

    for epoch in tqdm(range(args.epoch_1), desc="stage1_align"):
        align_loss_list = []
        enable_weighted_local_loss = epoch >= stage1_local_weight_start_epoch
        for image_features, patch_feature, rgb_render_image_features, rgb_patch_feature, s in tqdm(stage1_cache, leave=False):
            image_features = image_features.to(device)
            patch_feature = patch_feature.to(device)
            rgb_render_image_features = rgb_render_image_features.to(device)
            rgb_patch_feature = rgb_patch_feature.to(device)
            s = s.to(device)

            aligned_global, aligned_local = aligner(image_features, patch_feature)
            aligned_global = aligned_global / aligned_global.norm(dim=-1, keepdim=True)
            aligned_local = aligned_local / aligned_local.norm(dim=-1, keepdim=True)

            align_loss_global = 1 - metric(aligned_global, rgb_render_image_features).mean()

            align_local_loss = 1 - metric(aligned_local, rgb_patch_feature)

            if enable_weighted_local_loss:
                # Weighted local alignment loss per patch:
                weight = 1.0 + (args.align_local_weight_lambda * torch.sigmoid(s))
                align_loss_local = (weight * align_local_loss).mean()
            else:
                # Original local alignment loss (no weighting)
                align_loss_local = align_local_loss.mean()
            align_loss = align_loss_global + align_loss_local

            optimizer_align.zero_grad()
            align_loss.backward()
            optimizer_align.step()

            align_loss_list.append(align_loss.item())

        if (epoch + 1) % args.print_freq == 0:
            logger.info('stage1 epoch [{}/{}], align_loss:{:.4f}'.format(epoch + 1, args.epoch_1, np.mean(align_loss_list)))

    # Stage 2: optimize total_loss only (no align_loss optimization).
    rgb_prompt_learner.train()
    prompt_learner.train()
    aligner.eval()
    for p in aligner.parameters():
        p.requires_grad = False

    # In stage 2, enable alignment loss only for the last `epoch_2_alignment_loss` epochs.
    # Example: epoch_2=15, epoch_2_alignment_loss=5 -> start from epoch=10 (0-based): 10..14.
    stage2_alignment_start_epoch = max(0, args.epoch_2 - args.epoch_2_alignment_loss)

    for epoch in tqdm(range(args.epoch_2), desc="stage2_total"):
        rgb_d2_pixel_loss_list = []
        rgb_d3_global_loss_list = []
        rgb_d3_point_loss_list = []
        rgb_d2_image_loss_list = []

        alignment_loss_rgb_list = []
        alignment_loss_render_list = []

        d2_pixel_loss_list = []
        d3_global_loss_list = []
        d3_point_loss_list = []
        d2_image_loss_list = []

        enable_alignment_loss = epoch >= stage2_alignment_start_epoch

        for items in tqdm(train_dataloader, leave=False):
            label =  items['anomaly']
            render_image = items['d2_render_img'].to(device) # shape: (b, nv, c, h, w)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w) # shape: (b*nv, c, h, w)
            d2_render_anomaly = items['d2_render_anomaly'].to(device)
            d2_3d_cor = items['d2_3d_cor'].to(device)
            non_zero_index = items['non_zero_index'].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            render_gt_mask = items['d2_render_gt'].to(device)
            render_gt_mask[render_gt_mask > 0.5], render_gt_mask[render_gt_mask <= 0.5] = 1, 0
            render_gt_mask = render_gt_mask.reshape(-1, 1, h, w)

            with torch.no_grad():
                image_features, patch_features = model.encode_image(render_image, args.features_list, DPAM_layer = 20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_feature = patch_features[0]
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)

                aligned_global, aligned_local = aligner(image_features, patch_feature)
                aligned_global = aligned_global / aligned_global.norm(dim=-1, keepdim=True)
                aligned_local = aligned_local / aligned_local.norm(dim=-1, keepdim=True)

            # Compute both modalities' prompt features first (normal/anomaly for each).
            rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text = rgb_prompt_learner(cls_id=None)
            rgb_text_features = model.encode_text_learn(rgb_prompts, rgb_tokenized_prompts, rgb_compound_prompts_text).float()
            rgb_text_features = torch.stack(torch.chunk(rgb_text_features, dim=0, chunks=2), dim=1)
            rgb_text_features = rgb_text_features / rgb_text_features.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Contrastive alignment loss (4 anchors):
            # pseudo-RGB normal/anomaly align to rendering normal/anomaly (cross-modal same-type close)
            # and push away from within-modality opposite-type (same-modal different-type far).
            rgb_g_n = rgb_text_features[:, 0, :]
            rgb_g_a = rgb_text_features[:, 1, :]
            ren_g_n = text_features[:, 0, :]
            ren_g_a = text_features[:, 1, :]

            # Keep two-modality optimization separated:
            # - pseudo-RGB update: detach rendering prompts so gradients only flow to rgb_prompt_learner
            # - rendering update: detach pseudo-RGB prompts so gradients only flow to prompt_learner
            if enable_alignment_loss:
                alignment_loss_rgb = (
                    _contrastive_two_logits(rgb_g_n, ren_g_n.detach(), rgb_g_a)
                    + _contrastive_two_logits(rgb_g_a, ren_g_a.detach(), rgb_g_n)
                ) / 2.0

                alignment_loss_render = (
                    + _contrastive_two_logits(ren_g_n, rgb_g_n.detach(), ren_g_a)
                    + _contrastive_two_logits(ren_g_a, rgb_g_a.detach(), ren_g_n)
                ) / 2.0
            else:
                alignment_loss_rgb = torch.zeros((), device=device)
                alignment_loss_render = torch.zeros((), device=device)

            alignment_loss_rgb_list.append(alignment_loss_rgb.detach().item())
            alignment_loss_render_list.append(alignment_loss_render.detach().item())

            # Aligned feature-based prompt learning and anomaly detection losses.
            rgb_text_probs = aligned_global.unsqueeze(1) @ rgb_text_features.permute(0, 2, 1)
            rgb_text_probs = rgb_text_probs[:, 0, ...]/0.07
            d2_render_anomaly = d2_render_anomaly.reshape(-1,)
            rgb_d2_image_loss = F.cross_entropy(rgb_text_probs.squeeze(), d2_render_anomaly.long().cuda())

            rgb_text_probs = torch.chunk(rgb_text_probs,  nv, dim = 0)
            rgb_text_probs = torch.stack(rgb_text_probs, dim = 1).mean(1)
            rgb_d3_global_loss = F.cross_entropy(rgb_text_probs, label.long().cuda())

            rgb_similarity, _ = AnomalyCLIP_lib.compute_similarity(aligned_local, rgb_text_features[0])
            rgb_similarity_map = AnomalyCLIP_lib.get_similarity_map(rgb_similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
            rgb_d3_similarity_map = back_to_3d(rgb_similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)

            rgb_d3_point_loss = 0
            rgb_d3_point_loss += loss_dice(rgb_d3_similarity_map[:, 1, :, :], gt)
            rgb_d3_point_loss += loss_dice(rgb_d3_similarity_map[:, 0, :, :], 1-gt)

            rgb_d2_pixel_loss = 0
            rgb_d2_pixel_loss += loss_focal(rgb_similarity_map, render_gt_mask)
            rgb_d2_pixel_loss += loss_dice(rgb_similarity_map[:, 1, :, :], render_gt_mask)
            rgb_d2_pixel_loss += loss_dice(rgb_similarity_map[:, 0, :, :], 1-render_gt_mask)

            optimizer_rgb_total.zero_grad()
            total_rgb_loss = rgb_d3_point_loss + rgb_d2_pixel_loss + rgb_d3_global_loss + rgb_d2_image_loss
            if enable_alignment_loss:
                total_rgb_loss = total_rgb_loss + (args.alignment_loss_scale * alignment_loss_rgb)
            total_rgb_loss.backward()
            optimizer_rgb_total.step()
            rgb_d2_pixel_loss_list.append(rgb_d2_pixel_loss.item())
            rgb_d3_point_loss_list.append(rgb_d3_point_loss.item())
            rgb_d3_global_loss_list.append(rgb_d3_global_loss.item())
            rgb_d2_image_loss_list.append(rgb_d2_image_loss.item())
            # Rendering feature-based prompt learning and anomaly detection losses.
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...]/0.07
            # d2_render_anomaly = d2_render_anomaly.reshape(-1,)
            d2_image_loss = F.cross_entropy(text_probs.squeeze(), d2_render_anomaly.long().cuda())

            text_probs = torch.chunk(text_probs,  nv, dim = 0)
            text_probs = torch.stack(text_probs, dim = 1).mean(1)
            d3_global_loss = F.cross_entropy(text_probs, label.long().cuda())

            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)
            d3_similarity_map = back_to_3d(similarity_map.permute(0, 2, 3, 1), d2_3d_cor, non_zero_index).permute(0, 3, 1, 2)

            d3_point_loss = 0
            d3_point_loss += loss_dice(d3_similarity_map[:, 1, :, :], gt)
            d3_point_loss += loss_dice(d3_similarity_map[:, 0, :, :], 1-gt)

            d2_pixel_loss = 0
            d2_pixel_loss += loss_focal(similarity_map, render_gt_mask)
            d2_pixel_loss += loss_dice(similarity_map[:, 1, :, :], render_gt_mask)
            d2_pixel_loss += loss_dice(similarity_map[:, 0, :, :], 1-render_gt_mask)

            optimizer_total.zero_grad()
            total_loss = d3_point_loss + d2_pixel_loss + d3_global_loss + d2_image_loss
            if enable_alignment_loss:
                total_loss = total_loss + (args.alignment_loss_scale * alignment_loss_render)
            total_loss.backward()
            optimizer_total.step()
            d2_pixel_loss_list.append(d2_pixel_loss.item())
            d3_point_loss_list.append(d3_point_loss.item())
            d3_global_loss_list.append(d3_global_loss.item())
            d2_image_loss_list.append(d2_image_loss.item())

        if (epoch + 1) % args.print_freq == 0:
            logger.info('stage2-rgb epoch [{}/{}], d2_pixel_loss:{:.4f}, d3_point_loss:{:.4f}, d3_global_loss:{:.4f}, d2_image_loss:{:.4f}, alignment_loss:{:.6f}'.format(
                epoch + 1, args.epoch_2,
                np.mean(rgb_d2_pixel_loss_list), np.mean(rgb_d3_point_loss_list), np.mean(rgb_d3_global_loss_list), np.mean(rgb_d2_image_loss_list),
                np.mean(alignment_loss_rgb_list) if len(alignment_loss_rgb_list) > 0 else 0.0,
            ))
            logger.info('stage2-rendering epoch [{}/{}], d2_pixel_loss:{:.4f}, d3_point_loss:{:.4f}, d3_global_loss:{:.4f}, d2_image_loss:{:.4f}, alignment_loss:{:.6f}'.format(
                epoch + 1, args.epoch_2,
                np.mean(d2_pixel_loss_list), np.mean(d3_point_loss_list), np.mean(d3_global_loss_list), np.mean(d2_image_loss_list),
                np.mean(alignment_loss_render_list) if len(alignment_loss_render_list) > 0 else 0.0,
            ))

        if (epoch + 1) % args.save_freq == 0:
            rgb_ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '_rgb.pth')
            rendering_ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '_rendering.pth')
            torch.save({"rgb_prompt_learner": rgb_prompt_learner.state_dict(), "aligner": aligner.state_dict()}, rgb_ckp_path)
            torch.save({"prompt_learner": prompt_learner.state_dict()}, rendering_ckp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Align3D-AD", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint', help='path to save results')
    # parser.add_argument("--device", type=str, default='cuda:1', help="device")

    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--train_dataset_name", type=str, default='cookie', help="save frequency")

    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    # hyperparameter
    parser.add_argument("--alignment_loss_scale", type=float, default=0.1, help="scale for contrastive alignment loss")
    parser.add_argument("--align_local_weight_lambda", type=float, default=1.0, help="lambda for per-patch weighted local align loss in stage 1")
    parser.add_argument("--epoch_1", type=int, default=250, help="epochs for align_loss-only training")
    parser.add_argument("--epoch_1_local_loss", type=int, default=50, help="epochs for align_local_loss training")
    parser.add_argument("--epoch_2", type=int, default=15, help="epochs for total_loss-only training")
    parser.add_argument("--epoch_2_alignment_loss", type=int, default=5, help="epochs for alignment_loss training in stage 2")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")



    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
