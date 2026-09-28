# Align3D-AD
Official code for paper "Align3D-AD: Cross-Modal Feature Alignment and Dual-Prompt Learning for Zero-shot 3D Anomaly Detection"

## Introduction
Zero-shot 3D anomaly detection aims to identify anomalies in unseen categories without accessing their training data. Existing methods typically project 3D point clouds into multi-view renderings and process them with RGB-pretrained vision encoders, resulting in a domain gap between geometric renderings and RGB semantics. To address this issue, we propose Align3D-AD, a two-stage framework that leverages RGB observations from auxiliary categories as cross-modal guidance. In the first stage, Cross-Modal Feature Alignment maps rendering features into the RGB semantic space, refined by Semantic Consistency Reweighting (SCR). In the second stage, Modality-Aware Prompt Learning learns separate normal and anomalous prompts for RGB-aligned and rendering features, enhanced by Dual-Prompt Contrastive Alignment (DpCA). During inference, predictions from both branches are integrated using rendering observations only. Experiments on MVTec3D-AD, Eyecandies, and Real3D-AD demonstrate the effectiveness of Align3D-AD under one-vs-rest and cross-dataset evaluation settings.

## Overview
The overall framework of Align3D-AD is illustrated in the following figure.

![Overview](./Overall_Pipeline.png)
