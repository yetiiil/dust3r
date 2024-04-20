import argparse
import os
import torch
import numpy as np
import tempfile
import functools
import trimesh
from scipy.spatial.transform import Rotation

from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image 
import torch.nn.functional as F

import open3d as o3d

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import pandas as pd
import matplotlib.pyplot as pl

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1

ade20K = pd.read_csv("/scratch2/yuxili/interiorDesign/color_coding_semantic_segmentation_classes - Sheet1.csv")

class PointCloud:
    def __init__(self, point_cloud_path):
        pcd = o3d.io.read_point_cloud(point_cloud_path)
        self.points = np.asarray(pcd.points)
        self.num_points = self.points.shape[0]
        self.colors = np.asarray(pcd.colors)

    def get_homogeneous_coordinates(self):
        return np.append(self.points, np.ones((self.num_points, 1)), axis=-1)
    
def _convert_scene_output_to_glb(
    outdir,
    out_name,
    seg_name,
    image_path,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    intrinsics,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    clean_scene=False,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    intrinsics = to_numpy(intrinsics)

    # scene = trimesh.Scene()

    # full pointcloud
    # if as_pointcloud:
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    
    if not clean_scene:
        pct.export(outdir + out_name)
        tmp1 = PointCloud(outdir + out_name)
        
        poses = np.eye(4)
        print(poses)
        X = tmp1.get_homogeneous_coordinates()
        n_points = tmp1.num_points        
        intrinsics1 = np.zeros((4, 4))
        intrinsics1[:3, :3] = intrinsics[0]
        intrinsics1[3, 3] = 1

        intrinsics2 = np.zeros((4, 4))
        intrinsics2[:3, :3] = intrinsics[1]
        intrinsics2[3, 3] = 1

        intrinsic = (intrinsics1 + intrinsics2) / 2
        
        projected_points = np.zeros((n_points, 2), dtype=int)
        print("[INFO] Computing the visible points in each view.")

        # *******************************************************************************************************************
        # STEP 1: get the projected points
        # Get the coordinates of the projected points in the i-th view (i.e. the view with index idx)
        projected_points_not_norm = (intrinsic @ poses @ X.T).T
        # Get the mask of the points which have a non-null third coordinate to avoid division by zero
        mask = (
            projected_points_not_norm[:, 2] != 0
        )

        # don't do the division for point with the third coord equal to zero
        # Get non homogeneous coordinates of valid points (2D in the image)

        projected_points[mask] = np.column_stack(
            [
                [
                    projected_points_not_norm[:, 0][mask]
                    / projected_points_not_norm[:, 2][mask],
                    projected_points_not_norm[:, 1][mask]
                    / projected_points_not_norm[:, 2][mask],
                ]
            ]
        ).T
        
        # print(projected_points[:, 0].min())
        # print(projected_points[:, 0].max())
        # print(projected_points[:, 1].min())
        # print(projected_points[:, 1].max())
        
        image = Image.open(image_path)
        #resize image
        cache_dir = "/scratch2/yuxili/interiorDesign/huggingface/"

        processor = OneFormerProcessor.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large", cache_dir=cache_dir
        )
        model = OneFormerForUniversalSegmentation.from_pretrained(
            "shi-labs/oneformer_ade20k_swin_large", cache_dir=cache_dir
        )

        # Semantic Segmentation
        semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        semantic_outputs = model(**semantic_inputs)
        # pass through image_processor for postprocessing
        predicted_semantic_map = processor.post_process_semantic_segmentation(
            semantic_outputs, target_sizes=[image.size[::-1]]
        )[0]

        predicted_semantic_map = predicted_semantic_map.float()  # Convert tensor to float
        predicted_semantic_map = predicted_semantic_map.unsqueeze(0).unsqueeze(0) 
        downsampled_map = F.interpolate(predicted_semantic_map, size=(512, 512), mode='bilinear', align_corners=False)
        predicted_semantic_map = downsampled_map.squeeze(0).squeeze(0).cpu().detach().numpy()
        pc_color = tmp1.colors

        for idx, point in enumerate(projected_points):
            if point[0] < 0 :
                point[0] = 0
            if point[0] >= 512:
                point[0] = 511
            if point[1] < 0:
                point[1] = 0
            if point[1] >= 512:
                point[1] = 511
                
            color_index = int(predicted_semantic_map.T[point[0], point[1]])
            tmp_color = ade20K.iloc[color_index]["Color_Code (R,G,B)"].strip('()').split(',')  
            tmp_color  = [int(element) for element in tmp_color]
            pc_color[idx] = np.array(tmp_color)
        # tmp1.colors = pc_color
        # # Assuming 'points' and 'colors' are your arrays containing the point coordinates and colors
        # points = np.asarray(tmp1.points)
        # colors = np.asarray(tmp1.colors) / tmp1.colors.max()

        # # Create a PointCloud object
        # pcd = o3d.geometry.PointCloud()

        # # Assign the points and colors to the PointCloud object
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)

        # # Save the point cloud to a file
        # o3d.io.write_point_cloud("output1.ply", pcd)
            
    rot = np.eye(4)
    # Scale the object by 15 times in all directions
    scale_factor = 5 / (pts[:, 0].max() - pts[:, 0].min())
    # Scaling is applied directly to the diagonal elements of the rotation matrix
    # to maintain the rotation while scaling the object
    rot[0, 0] *= scale_factor
    rot[1, 1] *= scale_factor
    rot[2, 2] *= scale_factor
    pct.apply_transform(rot)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    rot_x = Rotation.from_euler("x", np.deg2rad(-90)).as_matrix()
    rot[:3, :3] = np.dot(rot[:3, :3], rot_x)
    rot_z = Rotation.from_euler("z", np.deg2rad(180)).as_matrix()
    rot[:3, :3] = np.dot(rot[:3, :3], rot_z)
    pct.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))

    o3d_pc = o3d.geometry.PointCloud()
    points = pct.vertices
    # Calculate the 25th percentile of the y coordinates (heights)
    percentile_30 = np.percentile(points[:, 2], 35)
    percentile_70 = np.percentile(points[:, 2], 80)

    # Filter the points that are higher than the 25th percentile
    o3d_pc.points = o3d.utility.Vector3dVector(points[(points[:, 2] > percentile_30)&(points[:, 2] < percentile_70)])
    # o3d.io.write_point_cloud("o3d_pc.ply", o3d_pc)

    # Use RANSAC to estimate the plane model
    plane_model, inliers = o3d_pc.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    # plane_model: [a, b, c, d] of the plane equation ax + by + cz + d = 0
    [a, b, c, d] = plane_model
    print("Plane model: [a, b, c, d] = ", plane_model)

    # Calculating angles
    theta_x_rad = np.arccos(a / np.sqrt(a**2 + b**2))
    # Converting radians to degrees for easier interpretation
    theta_x_deg = np.degrees(theta_x_rad)
    print("Angle in degrees: ", theta_x_deg)
    
    if theta_x_deg > 45:
        theta_x_deg = 90 - theta_x_deg
    
    rot = np.eye(4)
    rot_z = Rotation.from_euler("z", np.deg2rad(theta_x_deg)).as_matrix()
    rot[:3, :3] = np.dot(rot[:3, :3], rot_z)
    pct.apply_transform(rot)
    pct.export(outdir + out_name)
    
    if not clean_scene:
        tmp2 = PointCloud(outdir + out_name)
        tmp2.colors = pc_color
        
        # Assuming 'points' and 'colors' are your arrays containing the point coordinates and colors
        points = np.asarray(tmp2.points)
        colors = np.asarray(tmp2.colors) / 255

        # Create a PointCloud object
        pcd = o3d.geometry.PointCloud()

        # Assign the points and colors to the PointCloud object
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save the point cloud to a file
        o3d.io.write_point_cloud(outdir + seg_name, pcd)
    
    # scene.add_geometry(pct)
    # else:
    #     meshes = []
    #     for i in range(len(imgs)):
    #         meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
    #     mesh = trimesh.Trimesh(**cat_meshes(meshes))
    #     scene.add_geometry(mesh)

    # # add each camera
    # for i, pose_c2w in enumerate(cams2world):
    #     if isinstance(cam_color, list):
    #         camera_edge_color = cam_color[i]
    #     else:
    #         camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
    #     add_scene_cam(
    #         scene,
    #         pose_c2w,
    #         camera_edge_color,
    #         None if transparent_cams else imgs[i],
    #         focals[i],
    #         imsize=imgs[i].shape[1::-1],
    #         screen_width=cam_size,
    #     )

    # rot = np.eye(4)
    # rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    # scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    # outfile = os.path.join(outdir, 'scene.glb')
    # print('(exporting 3D scene to', outfile, ')')
    # scene.export(file_obj=outfile)
    # return outfile


def get_3D_model_from_scene(
    outdir,
    out_name,
    seg_name,
    image_path,
    scene,
    min_conf_thr=3,
    as_pointcloud=False,
    mask_sky=False,
    clean_depth=False,
    transparent_cams=False,
    cam_size=0.05,
    clean_scene=False,
):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    intrinsic = scene.get_intrinsics().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(
        outdir,
        out_name,
        seg_name,
        image_path,
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        intrinsic,
        as_pointcloud=as_pointcloud,
        transparent_cams=transparent_cams,
        cam_size=cam_size,
        clean_scene=clean_scene,
    )


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="out directory")
    parser.add_argument("--image_path", type=str, default="/scratch2/yuxili/interiorDesign/output/bedroom/2024-04-02-13-25-53-experiment/bedroom.png", help="image path")
    parser.add_argument("--out_name", type=str, help="out name")
    parser.add_argument("--clean_scene", type=bool, default=False, help="clean scene")
    parser.add_argument("--seg_name", type=str, default="", help="segmentation name")
    return parser


parser = get_args_parser()
args = parser.parse_args()

out_dir = args.out_dir
clean_scene = args.clean_scene
image_path = args.image_path
out_name = args.out_name
seg_name = args.seg_name

model_path = "/scratch2/yuxili/interiorDesign/dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
device = "cuda"
batch_size = 1
schedule = "cosine"
lr = 0.01
niter = 300
model = load_model(model_path, device)
# load_images can take a list of images or a directory
images = load_images([image_path, image_path], size=512)
pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
output = inference(pairs, model, device, batch_size=batch_size)

scene = global_aligner(
    output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
)
loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

get_3D_model_from_scene(
    out_dir,
    out_name,
    seg_name,
    image_path,
    scene,
    min_conf_thr=3,
    as_pointcloud=True,
    mask_sky=False,
    clean_depth=True,
    transparent_cams=False,
    cam_size=0.05,
    clean_scene=clean_scene,
)
    
