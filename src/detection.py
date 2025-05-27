import matplotlib.pyplot as plt
import os 
import cv2
import numpy as np
from tqdm import tqdm
import faiss
import tifffile as tiff
import time
import torch
import json
from sklearn.cluster import KMeans
from src.utils import augment_image, dists2map, min_max_norm, cvt2heatmap, heatmap_on_image
from src.post_eval import mean_top1p
from src.sampler import GreedyCoresetSampler

def fill_closed_regions(image):
    if image is None:
        print("Invalid input image")
        return None

    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled = np.zeros_like(image)
    cv2.drawContours(filled, contours, contourIdx=-1, color=255, thickness=cv2.FILLED)

    return filled

def run_anomaly_detection_multilayer(
        model,
        object_name,
        data_root,
        n_ref_samples,
        object_anomalies,
        plots_dir,
        device,
        save_examples=False,
        masking=None,
        mask_ref_images=False,
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_patch_dists=True,
        save_tiffs=False):
    """
    Updated to support multi-layer feature extraction and layer-wise knn matching.
    """

    assert knn_metric in ["L2", "L2_normalized"]
    type_anomalies = list(set(object_anomalies[object_name] + ['good']))

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    img_ref_samples = sorted(os.listdir(img_ref_folder))

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
        n_ref_samples = len(img_ref_samples)
        
    ######################################## Random selection ########################################    
    # if n_ref_samples != -1:
    #     img_ref_samples = img_ref_samples[seed * n_ref_samples:(seed + 1) * n_ref_samples]

    ####################################### Coreset selection ########################################   
    # Extract CLS features for all reference images
    cls_features = []
    valid_img_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img_tensor, _ = model.prepare_image(img_rgb)
            cls_feats = model.extract_cls_features(img_tensor)
            cls_features.append(cls_feats.squeeze().cpu())
            valid_img_names.append(img_name)

    cls_features = torch.stack(cls_features).to(device)  # Shape: (n_samples, 1024)
    sampler = GreedyCoresetSampler(percentage=0.1, device=device, dimension_to_project_features_to=1024)
    selected_indices = sampler.run(cls_features)
    
    # Select the corresponding image names
    img_ref_samples = [valid_img_names[idx] for idx in selected_indices]

    ######################################## K-Means selection ########################################
    # # Extract CLS features for all reference images
    # cls_features = []
    # valid_img_names = []
    # with torch.inference_mode():
    #     for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
    #         image_path = os.path.join(img_ref_folder, img_name)
    #         img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #         img_tensor, _ = model.prepare_image(img_rgb)
    #         cls_feats = model.extract_cls_features(img_tensor)
    #         cls_features.append(cls_feats.cpu().numpy().squeeze())
    #         valid_img_names.append(img_name)

    # cls_features = np.array(cls_features)  # Shape: (n_samples, 1024)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=n_ref_samples, random_state=seed, n_init=10)
    # kmeans.fit(cls_features)

    # # Find the image closest to each cluster center
    # cluster_centers = kmeans.cluster_centers_
    # selected_indices = []
    # for center in cluster_centers:
    #     distances = np.linalg.norm(cls_features - center, axis=1)
    #     closest_idx = np.argmin(distances)
    #     selected_indices.append(closest_idx)

    # # Select the corresponding image names
    # img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    ####################################################################################################
    
    
    feature_refs = {}  # {layer_name: [features]}
    knn_indices = {}   # {layer_name: faiss_index}
    grid_size = None

    with torch.inference_mode():
        start_time = time.time()

        for img_name in tqdm(img_ref_samples, desc="Extracting reference features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            aug_images = augment_image(img_rgb) if rotation else [img_rgb]
            for aug in aug_images:
                img_tensor, grid_size = model.prepare_image(aug)
                feats_dict = model.extract_features(img_tensor)
                idx=0
                mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1,
                                                         masking_type=(mask_ref_images and masking))
                for feats in feats_dict:
                    selected_feats = feats[mask]
                    if f'layer{idx}' not in feature_refs:
                        feature_refs[f'layer{idx}'] = []
                    feature_refs[f'layer{idx}'].append(selected_feats)
                    idx+=1

        # Concatenate and build FAISS index for each layer
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype('float32')
            if knn_metric == 'L2_normalized':
                faiss.normalize_L2(layer_feats)

            if faiss_on_cpu:
                index = faiss.IndexFlatL2(layer_feats.shape[1])
            else:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, layer_feats.shape[1])

            index.add(layer_feats)
            knn_indices[layer_name] = index

        time_memorybank = time.time() - start_time

        inference_times = {}
        anomaly_scores = {}

        for anomaly_type in tqdm(type_anomalies, desc=f"Processing {object_name}"):
            test_dir = f"{data_root}/{object_name}/test_public/{anomaly_type}"
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}", exist_ok=True)
            os.makedirs(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}", exist_ok=True)

            for idx, test_img_name in enumerate(sorted(os.listdir(test_dir))):
                start_time = time.time()
                test_path = os.path.join(test_dir, test_img_name)
                
                img_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)
                img_tensor, _ = model.prepare_image(img_rgb)
                feats_dict = model.extract_features(img_tensor)

                dists_per_layer = []
                
                mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1, masking_type=masking)
                for num, feats in enumerate(feats_dict):
                    masked_feats = feats[mask]

                    if knn_metric == "L2_normalized":
                        faiss.normalize_L2(masked_feats)

                    dists, _ = knn_indices[f'layer{num}'].search(masked_feats, k=knn_neighbors)
                    if knn_neighbors > 1:
                        dists = dists.mean(axis=1)

                    dists = 1 - dists  # cosine distance

                    dmap = np.zeros_like(mask, dtype=float)
                    dmap[mask] = dists.squeeze()
                    dmap = dmap.reshape(grid_size)

                    dmap_resized = cv2.resize(dmap, (img_rgb.shape[1], img_rgb.shape[0]))
                    dists_per_layer.append(dmap_resized)

                # Average the 4 layers
                anomaly_map = np.mean(dists_per_layer, axis=0)
                anomaly_map_norm = min_max_norm(anomaly_map)
                score = mean_top1p(anomaly_map.flatten())

                inference_times[f"{anomaly_type}/{test_img_name}"] = time.time() - start_time
                anomaly_scores[f"{anomaly_type}/{test_img_name}"] = score

                if save_tiffs:
                    heatmap = cvt2heatmap(anomaly_map_norm * 255)
                    hm_on_img = heatmap_on_image(heatmap, img_rgb)
                    fname = os.path.splitext(test_img_name)[0]
                    cv2.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test_hm_on_img/{anomaly_type}/{fname}.jpg", hm_on_img)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{fname}.tiff", anomaly_map)
                if save_patch_dists:
                    np.save(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{anomaly_type}/{test_img_name.split('.')[0]}.npy", anomaly_map)

                if save_examples and idx < 3:
                    num_layers = len(dists_per_layer)
                    cols = 3
                    rows = int(np.ceil((3 + num_layers) / cols))

                    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
                    axes = axes.flatten()

                    # 原始图像
                    axes[0].imshow(img_rgb)
                    axes[0].set_title("Test Image")
                    axes[0].axis("off")

                    # 平均 anomaly map
                    im = axes[1].imshow(anomaly_map_norm, cmap='jet')
                    axes[1].set_title("Avg Anomaly Map")
                    axes[1].axis("off")
                    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, orientation="horizontal")

                    # Histogram
                    axes[2].hist(anomaly_map.flatten(), bins=50)
                    axes[2].axvline(score, color='red', linestyle='dashed')
                    axes[2].set_title("Score Histogram")

                    # heatmaps by layers
                    for i, d_masked in enumerate(dists_per_layer):
                        ax = axes[3 + i]
                        im = ax.imshow(d_masked, cmap='jet')
                        ax.set_title(f"Layer {i} Map")
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, orientation="horizontal")

                    for j in range(3 + num_layers, len(axes)):
                        axes[j].axis("off")

                    plt.tight_layout()
                    example_dir = f"{plots_dir}/{object_name}/examples"
                    os.makedirs(example_dir, exist_ok=True)
                    plt.savefig(f"{example_dir}/example_{anomaly_type}_{idx}.png")
                    plt.close()


    return anomaly_scores, time_memorybank, inference_times

def run_anomaly_detection_multilayer_private(
        model,
        object_name,
        data_root,
        n_ref_samples,
        plots_dir,
        save_examples=False,
        masking=None,
        mask_ref_images=False,
        rotation=False,
        knn_metric='L2_normalized',
        knn_neighbors=1,
        faiss_on_cpu=False,
        seed=0,
        save_tiffs=False):
    """
    Updated to support multi-layer feature extraction and layer-wise knn matching.
    """

    assert knn_metric in ["L2", "L2_normalized"]

    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    img_ref_samples = sorted(os.listdir(img_ref_folder))
    # if n_ref_samples != -1:
    #     img_ref_samples = img_ref_samples[seed * n_ref_samples:(seed + 1) * n_ref_samples]

    if len(img_ref_samples) < n_ref_samples:
        print(f"Warning: Not enough reference samples for {object_name}! Only {len(img_ref_samples)} samples available.")
        n_ref_samples = len(img_ref_samples)
        
    ######################################## Coreset selection ########################################   
    # Extract CLS features for all reference images
    cls_features = []
    valid_img_names = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            img_tensor, _ = model.prepare_image(img_rgb)
            cls_feats = model.extract_cls_features(img_tensor)
            cls_features.append(cls_feats.squeeze().cpu())
            valid_img_names.append(img_name)

    cls_features = torch.stack(cls_features).to(device)  # Shape: (n_samples, 1024)
    sampler = GreedyCoresetSampler(percentage=0.1, device=device, dimension_to_project_features_to=1024)
    selected_indices = sampler.run(cls_features)
    
    # Select the corresponding image names
    img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    
    ######################################## K-Means selection ########################################
    # # Extract CLS features for all reference images
    # cls_features = []
    # valid_img_names = []
    # with torch.inference_mode():
    #     for img_name in tqdm(img_ref_samples, desc="Extracting CLS features", leave=False):
    #         image_path = os.path.join(img_ref_folder, img_name)
    #         img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #         img_tensor, _ = model.prepare_image(img_rgb)
    #         cls_feats = model.extract_cls_features(img_tensor)
    #         cls_features.append(cls_feats.cpu().numpy().squeeze())
    #         valid_img_names.append(img_name)

    # cls_features = np.array(cls_features)  # Shape: (n_samples, 1024)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=n_ref_samples, random_state=seed, n_init=10)
    # kmeans.fit(cls_features)

    # # Find the image closest to each cluster center
    # cluster_centers = kmeans.cluster_centers_
    # selected_indices = []
    # for center in cluster_centers:
    #     distances = np.linalg.norm(cls_features - center, axis=1)
    #     closest_idx = np.argmin(distances)
    #     selected_indices.append(closest_idx)

    # # Select the corresponding image names
    # img_ref_samples = [valid_img_names[idx] for idx in selected_indices]
    ####################################################################################################    
    

    feature_refs = {}  # {layer_name: [features]}
    knn_indices = {}   # {layer_name: faiss_index}
    grid_size = None

    with torch.inference_mode():
        start_time = time.time()

        for img_name in tqdm(img_ref_samples, desc="Extracting reference features", leave=False):
            image_path = os.path.join(img_ref_folder, img_name)
            img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            aug_images = augment_image(img_rgb) if rotation else [img_rgb]
            for aug in aug_images:
                img_tensor, grid_size = model.prepare_image(aug)
                feats_dict = model.extract_features(img_tensor)
                # cls_feats_dict = model.extract_cls_features(img_tensor)
                idx=0
                for feats in feats_dict:
                    mask = model.compute_background_mask(feats, grid_size, threshold=1,
                                                         masking_type=(mask_ref_images and masking))
                    selected_feats = feats[mask]
                    if f'layer{idx}' not in feature_refs:
                        feature_refs[f'layer{idx}'] = []
                    feature_refs[f'layer{idx}'].append(selected_feats)
                    idx+=1

        # Concatenate and build FAISS index for each layer
        for layer_name, feats_list in feature_refs.items():
            layer_feats = np.concatenate(feats_list, axis=0).astype('float32')
            if knn_metric == 'L2_normalized':
                faiss.normalize_L2(layer_feats)

            if faiss_on_cpu:
                index = faiss.IndexFlatL2(layer_feats.shape[1])
            else:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, layer_feats.shape[1])


            index.add(layer_feats)
            knn_indices[layer_name] = index

        time_memorybank = time.time() - start_time

        inference_times = {}
        anomaly_scores = {}
        print(f"processing test samples ({object_name})")
        for test_type in ["test_private", "test_private_mixed"]:
            data_dir = f"{data_root}/{object_name}/{test_type}"

            os.makedirs(f"{plots_dir}/submission_folder/anomaly_images/{object_name}/{test_type}", exist_ok=True)
            os.makedirs(f"{plots_dir}/submission_folder/anomaly_images_thresholded/{object_name}/{test_type}", exist_ok=True)

            for idx, test_img_name in enumerate(sorted(os.listdir(data_dir))):
                start_time = time.time()
                test_path = f"{data_dir}/{test_img_name}"
                img_rgb = cv2.cvtColor(cv2.imread(test_path), cv2.COLOR_BGR2RGB)

                img_tensor, grid_size2 = model.prepare_image(img_rgb)
                feats_dict = model.extract_features(img_tensor)

                dists_per_layer = []

                mask = model.compute_background_mask(feats_dict[0], grid_size, threshold=1, masking_type=masking)
                for num, feats in enumerate(feats_dict):
                    masked_feats = feats[mask]

                    if knn_metric == "L2_normalized":
                        faiss.normalize_L2(masked_feats)

                    dists, _ = knn_indices[f'layer{num}'].search(masked_feats, k=knn_neighbors)
                    if knn_neighbors > 1:
                        dists = dists.mean(axis=1)

                    dists = 1 - dists  # cosine distance

                    dmap = np.zeros_like(mask, dtype=float)
                    dmap[mask] = dists.squeeze()
                    dmap = dmap.reshape(grid_size2)

                    dists_per_layer.append(dmap)

                # Average the 4 layers
                anomaly_map = np.mean(dists_per_layer, axis=0)
                score = mean_top1p(anomaly_map.flatten())

                inference_times[f"{test_img_name}"] = time.time() - start_time
                anomaly_scores[f"{test_img_name}"] = score

                test_img_name = test_img_name.split(".")[0]

                with open("./results_dir/metrics_seed=0.json", "r") as f:
                    object_thresholds = json.load(f)

                if save_tiffs:
                    anomaly_map_f16 = anomaly_map.astype(np.float16)
                    tiff.imwrite(
                        f"{plots_dir}/submission_folder/anomaly_images/{object_name}/{test_type}/{test_img_name}.tiff",
                        anomaly_map_f16
                    )

                    full_res_map = dists2map(anomaly_map, img_rgb.shape)
                    threshold = object_thresholds[object_name]["best_thre"]
                    binary_mask = (full_res_map > threshold).astype(np.uint8) * 255
                    
                    fill_config = {
                        'can': False,
                        'fabric': True,
                        'fruit_jelly': False,
                        'rice': False,
                        'sheet_metal': False,
                        'vial': False,
                        'wallplugs': False,
                        'walnuts': True
                    }
                    
                    if fill_config[object_name]:
                        binary_mask = fill_closed_regions(binary_mask)
                    cv2.imwrite(
                        f"{plots_dir}/submission_folder/anomaly_images_thresholded/{object_name}/{test_type}/{test_img_name}.png",
                        binary_mask
                    )


    return anomaly_scores, time_memorybank, inference_times