import os.path as osp
from glob import glob
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm
import trimesh
import torch
import os

from contextlib import contextmanager
from functools import partial
import multiprocessing
import os
import signal
from typing import List, Union

import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm
import trimesh
import os
import torch

import signal
import imageio
import numpy as np

import bpy
from render import *
from PIL import Image

import os
os.environ['MUJOCO_GL'] = 'osmesa'  # 或 'osmesa'
def render_three(object_path: str,
                 out_dir: str,
                 base_id: str,
                 resolution: int = 224,
                 yaw0: float = 0,
                 pitch: float = 30,
                 radius: float = 2,
                 fov_deg: float = 45,
                 engine: str = "CYCLES",
                 geo_mode: bool = False):
    """
    yaw0, yaw0+120°, yaw0+240°。
    <id>_1.png, <id>_2.png, <id>_3.png
    """
    os.makedirs(out_dir, exist_ok=True)
    yaws = [yaw0, yaw0 + 120, yaw0 + 240]
    init_scene()
    init_render(engine=engine, resolution=resolution, geo_mode=geo_mode)
    init_nodes()                    
    # normalize_scene()
    init_lighting()
    load_object(object_path)
    if geo_mode:
        override_material()
    images = []
    for i, y in enumerate(yaws, start=1):
        out_png = os.path.join(out_dir, f"{base_id}_{i}.png")
        cam = init_camera()
        yaw_r, pitch_r = math.radians(y), math.radians(pitch)
        cam.location = (
            radius * math.cos(pitch_r) * math.cos(yaw_r),
            radius * math.sin(yaw_r) * math.cos(pitch_r),
            radius * math.sin(pitch_r)
        )
        cam.data.lens = 16 / math.tan(math.radians(fov_deg) / 2)
        cam.data.clip_start = 0.01
        cam.data.clip_end   = radius * 10
        bpy.context.scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        # print("Render finished:", out_png)
        images.append(Image.open(out_png))
        bpy.data.images.remove(bpy.data.images["Render Result"])
    # bpy.data.images["Render Result"].user_clear()
    return images
        

    
    
        
def quaternion_to_axis_angle(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    angle = 2 * np.arccos(w)
    sin_theta = np.sqrt(1 - w ** 2)  # sin(theta/2)
    
    zero_mask = sin_theta < 1e-6
    axis = np.zeros(q.shape[:-1] + (3,))
    
    axis = np.where(zero_mask[..., None], 
                    np.array([1., 0., 0.]), 
                    np.stack([x, y, z], axis=-1) / np.where(zero_mask[..., None], 1, sin_theta[..., None]))
    return axis, angle,q
def get_mujoco_info_from_mesh(mesh,pos=None,mass=None,inertia=None,path=None,int_v = None):
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces).astype(np.int32)

    asset_xml, body_xml = "", ""
    faces_str = "  ".join(f"{v1:d} {v2:d} {v3:d}" for v1, v2, v3 in faces)
    vertices_str = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    if type(int_v)!=type(None):
        asset_xml += f"""
                <mesh name="mesh_0" scale="1 1 1" vertex="{vertices_str}" face="{faces_str}" inertia="{int_v}"/>
            """
    else:
        asset_xml += f"""
                <mesh name="mesh_0" scale="1 1 1" vertex="{vertices_str}" face="{faces_str}"/>
            """
    
    """ "{com_pos_str}" mass="{mesh_masses[i]}" diaginertia="{inertia_str}"""
    if type(pos)!= type(None) and type(mass)!=type(None) and type(inertia)!=type(None):
        phy_xml = f"""<inertial pos="{pos[0]} {pos[1]} {pos[2]}" mass="{mass}"  fullinertia="{inertia[0,0]} {inertia[1,1]} {inertia[2,2]} {inertia[0,1]} {inertia[0,2]} {inertia[1,2]}" />"""
    else:
        phy_xml = ""
    body_xml += f"""
            <body name="mesh_body_0" pos="0 0 0">
                <joint name="free_joint_0" type="free" />
                {phy_xml}
                <geom name="mesh_geom_0" type="mesh" mesh="mesh_0" density = "1.0" rgba="0 1 1 1" />
            </body>
        """
    camera_xml = """
    <camera 
        name="fixed_cam" 
          pos="-1.882458 -0.639713 -0.217066"
    xyaxes="0.321757 -0.946822 0.000000  -0.102762 -0.034921 0.994093"
        fovy="60"/>
"""
    model_xml = f"""
        <mujoco model="mesh_simulation">
            <compiler 
            meshdir="../meshes_mujoco/" 
            balanceinertia="true" 
            discardvisual="false" />
            <asset>
                {asset_xml}
            </asset>

            <worldbody>
                <!-- Ground plane -->
                <light name="main_light" pos="0 0 4" dir="0 0 -1" directional="true" diffuse="1 1 1" specular="0.5 0.5 0.5"/>
                <geom name="ground_plane" type="plane" pos="0 0 0" size="0 0 1"  rgba="1 1 1 1"  />
                <!-- Mesh object with free joint --> 
                {body_xml}
                 {camera_xml}
            </worldbody>
        </mujoco>
    """
    if path:
        with open(path,"w") as f:
            f.write(model_xml)
    model = mujoco.MjModel.from_xml_string(model_xml)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mesh_body_0")
    mass = model.body_mass[body_id]
    com = model.body_ipos[body_id]
    # print("inertia ",)
    iquat = model.body_iquat[body_id]
    # print(iquat)
    inertia = model.body_inertia[body_id] 
    
    res = np.zeros((9, 1), dtype=np.float64)   
    mujoco.mju_quat2Mat(res, iquat)
    R_iq = res.reshape((3, 3))  
    orthogonality = R_iq.T @ R_iq

    I_full_body = R_iq @ np.diag(inertia) @ R_iq.T
    
    # print(I_full_body)
    data = mujoco.MjData(model)
    return model, data, mass, com, I_full_body

def get_sim_angles(
    model,data,
    timeout: float = 5.0,
    render=0
):      
    os.makedirs('simulation_images', exist_ok=True)
    
    def save_image(viewer, filename):
        img = viewer.read_pixels(camid=0)
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join('simulation_images', filename))
    
    if render == 1:
        viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen')
        save_image(viewer, 'initial_state.png')
        start_img = viewer.read_pixels(camid=0)  
        
        
    def simulate():
        
        duration = 10.0  # seconds
        model.opt.timestep = 0.001

        while data.time < duration:
            mujoco.mj_step(model, data)
            if render == 1 and abs(data.time - duration) <= 0.001:  
                save_image(viewer, 'final_state.png')
        rotation = [data.qpos[3:]]
        _, angles,q = quaternion_to_axis_angle(np.array(rotation))
        angles = np.rad2deg(angles)
        return angles[0],q[0]

    try:
        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutError("Simulation timed out")
            
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(int(timeout))
            
            try:
                yield
            finally:
                signal.alarm(0)
        
        with time_limit(timeout):
            return simulate()
    except TimeoutError:
        print("Simulation timed out")
        if render == 1:
            viewer.close()
        return None
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        if render == 1:
            viewer.close()
        return None
    
    
### The Chamfer Distance and F-score is drawn from DSO

def icp_align(source, target, max_iterations=50, tolerance=1e-6):
    source_pc = o3d.geometry.PointCloud()
    target_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source)
    target_pc.points = o3d.utility.Vector3dVector(target)

    # Apply ICP
    threshold = 0.02
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=tolerance)
    )

    # Apply the transformation
    aligned_source = np.asarray(source_pc.transform(icp_result.transformation).points)
    return aligned_source


def compute_cd_and_fscore(gen_model, gt_model, n_points=10000, tau=0.05):
    gen_points, _ = trimesh.sample.sample_surface(gen_model, n_points)
    gt_points, _ = trimesh.sample.sample_surface(gt_model, n_points)

    gen_points = icp_align(gen_points, gt_points)

    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(gen_points)

    pred_to_gt_distances, _ = gt_tree.query(gen_points)
    gt_to_pred_distances, _ = pred_tree.query(gt_points)

    precision = np.mean(pred_to_gt_distances < tau)
    recall = np.mean(gt_to_pred_distances < tau)

    if precision + recall == 0:
        fscore = 0
    else:
        fscore = 2 * (precision * recall) / (precision + recall)
    chamfer_distance = (np.mean(pred_to_gt_distances) + np.mean(gt_to_pred_distances)) / 2
    return fscore, chamfer_distance

def eval_mesh(surface,target=None,inner=None,input_condition = None, cache_dir = "./",render =0,device = "cuda",base_id = "0",render_mesh = None):
    """
    Input:  
            input_condition: PIL.Image (I-to-3D)
                             [PIL.Image] (mutil I-to-3D)
                             str (T-to-3D)
            surface: generated surface 
            target: target mesh (should be GT (generated by Trellis) for compute CF and F-score)
            inner: the inner volume mesh (generated by matching cubes) for Stand & Stability
            render: mode 0 (not render)  1 (render the begin frame and the end frame)
    """
    
    ## preprocess the input
    
    
    # I should first compute the Stand and Stability and the Volume (This process utilizes the mujoco)
    # I think the mesh has
    
    result = {}
    if type(inner) == type(None):
        model, data, l_mass, l_com, l_inertia = get_mujoco_info_from_mesh(surface,path = os.path.join(cache_dir,"mujoco.xml"))
        angle,q = get_sim_angles( model, data,timeout=25.0,render = render)
        result["volume"] = l_mass ## use mass to represent the volume (considering the rho = 1.0)
    else:
        # I should consider the obj have the inner mesh (mass center, mass, inertia changed)
        _,__,sur_mass, sur_com, sur_inertia = get_mujoco_info_from_mesh(surface)
        try:
            _,__,in_mass, in_com, in_inertia = get_mujoco_info_from_mesh(inner)
            assert sur_mass>in_mass
            left_mass = sur_mass-in_mass
        
            left_com = (sur_mass * sur_com - in_mass * in_com) / left_mass
            sur_inertia_to_origin = sur_inertia + sur_mass*(
                np.eye(3)*( sur_com[1]**2+sur_com[2]**2 +sur_com[0]**2)-
                sur_com.reshape(3,1)@(sur_com.reshape(1,3))
            )
            in_inertia_to_origin = in_inertia + in_mass*(
                np.eye(3)*( in_com[1]**2+in_com[2]**2 +in_com[0]**2)-
                in_com.reshape(3,1)@(in_com.reshape(1,3))
            )
            left_inertia_to_origin = sur_inertia_to_origin - in_inertia_to_origin
            left_inertia =  left_inertia_to_origin - (
                np.eye(3)*( left_com[1]**2+left_com[2]**2 +left_com[0]**2)-
                left_com.reshape(3,1)@(left_com.reshape(1,3))
            )*left_mass

            model, data, l_mass, l_com, l_inertia = get_mujoco_info_from_mesh(surface,
                                                                              pos=left_com,mass=left_mass,inertia=left_inertia,
                                                                              path = os.path.join(cache_dir,"mujoco.xml"))
        except:
            _,__,sur_mass, sur_com, sur_inertia = get_mujoco_info_from_mesh(surface)
            _,__,in_mass, in_com, in_inertia = get_mujoco_info_from_mesh(inner,int_v = "exact")
            left_mass = sur_mass-in_mass
        
            left_com = (sur_mass * sur_com - in_mass * in_com) / left_mass
            sur_inertia_to_origin = sur_inertia + sur_mass*(
                np.eye(3)*( sur_com[1]**2+sur_com[2]**2 +sur_com[0]**2)-
                sur_com.reshape(3,1)@(sur_com.reshape(1,3))
            )
            in_inertia_to_origin = in_inertia + in_mass*(
                np.eye(3)*( in_com[1]**2+in_com[2]**2 +in_com[0]**2)-
                in_com.reshape(3,1)@(in_com.reshape(1,3))
            )
            left_inertia_to_origin = sur_inertia_to_origin - in_inertia_to_origin
            left_inertia =  left_inertia_to_origin - (
                np.eye(3)*( left_com[1]**2+left_com[2]**2 +left_com[0]**2)-
                left_com.reshape(3,1)@(left_com.reshape(1,3))
            )*left_mass

            model, data, l_mass, l_com, l_inertia = get_mujoco_info_from_mesh(surface,
                                                                              pos=left_com,mass=left_mass,inertia=left_inertia,
                                                                              path = os.path.join(cache_dir,"mujoco.xml"))
        
        
        assert sur_mass>=0 and in_mass>=0
        
        angle,q = get_sim_angles(
            model, data,
            timeout=25.0,
            render =render
        )
        result["volume"] = l_mass
        assert abs(l_mass-left_mass)<0.0001
    result["stand"] = angle < 20
    result["rotation angle"] = angle
    result["q"] = q
    if type(target) != type(None):
        fscore,cd = compute_cd_and_fscore(surface,target)
        result["chamfer distance"] = cd
        result["f score"] = fscore
    
    if type(input_condition) != type(None):
        cache_path = os.path.join(cache_dir,"surface-for-render-images.glb")
        render_mesh.export(cache_path)
        if os.path.exists(os.path.join(cache_dir, f"{base_id}_1.png")):
            rendered_images = [
                Image.open(os.path.join(cache_dir, f"{base_id}_1.png")),
                Image.open(os.path.join(cache_dir, f"{base_id}_2.png")),
                Image.open(os.path.join(cache_dir, f"{base_id}_3.png"))
            ]
        else:
            rendered_images = render_three(cache_path,out_dir= cache_dir,base_id = base_id,)
        clipscore = compute_conds_images_clipsim(input_condition,rendered_images,device = device)
        result["clipscore"] = clipscore
    return result
def compute_conds_images_clipsim(conditions,generated_images,device = "cpu"):
    with torch.no_grad():
        if isinstance(conditions, (str, Image.Image)):
            conditions = [conditions]
        if isinstance(generated_images, Image.Image):
            generated_images = [generated_images]
        cond_inputs = []
        for cond in conditions:
            if isinstance(cond, str) or isinstance(cond, Image.Image):
                cond_inputs.append(cond)
            else:
                raise ValueError("condition error")
        gen_images_inputs = generated_images

        cond_features = []
        for cond in cond_inputs:
            if isinstance(cond, str):
                inputs = processor(text=[cond], return_tensors="pt", 
                                    padding=True, 
                                    truncation=True,  
                                    max_length=77,    
                                  )
                with torch.no_grad():
                    features = model.get_text_features(input_ids = inputs['input_ids'].to(device), attention_mask = inputs['attention_mask'].to(device))
                cond_features.append(features)
            else:
                inputs = processor(images=[cond], return_tensors="pt")
                
                with torch.no_grad():
                    features = model.get_image_features(pixel_values = inputs['pixel_values'].to(device))
                cond_features.append(features)
        gen_inputs = processor(images=gen_images_inputs, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_features = model.get_image_features(**gen_inputs)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)

        total_score = 0.0
        count = 0
        
        cond_feat = torch.cat(cond_features, dim=0)
        cond_feat = cond_feat / cond_feat.norm(dim=-1, keepdim=True)
        assert len(cond_feat.shape)==2 and (abs((cond_feat[0].norm()-1).item())<0.0001)
        
        similarity = (cond_feat @ gen_features.T) * 100  
        similarity = torch.where(similarity>=0,similarity, similarity*0.0)
        avg_score =  similarity.mean()

        return avg_score.item()
# Function to load and prepare a test sample
def get_test_sample(result_dir_path, cond, trellis_out_path=None, id_=0):
        test_sample = {}
        base_path = result_dir_path
        
        # Load surface mesh from STL file
        surface = trimesh.load(os.path.join(base_path,"final_surface.stl"), force="mesh")
        # Reorder vertices to match desired axis orientation
        surface.vertices = surface.vertices[:, [0,2,1]]
        # Move mesh to ground plane (Z min = 0)
        move_to_ground = surface.vertices[:, 2].min()
        surface.vertices[:, 2] -= move_to_ground
        
        test_sample["surface"] = surface.copy()
        
        # Try loading inner mesh if available
        try:
            inner = trimesh.load(os.path.join(base_path,f"final_inner.stl"), force="mesh")
            inner.vertices = inner.vertices[:, [0,2,1]]
            inner.vertices[:, 2] -= move_to_ground
            test_sample["inner"] = inner.copy()
        except:
            print("No inner surface!")  # Print message if inner mesh not found  
        # If a trellis/target mesh is provided, load it
        if trellis_out_path:
            target = trimesh.load(trellis_out_path, force="mesh")
            target.vertices = target.vertices[:, [0,2,1]]
            move_to_ground = target.vertices[:, 2].min()
            target.vertices[:, 2] -= move_to_ground
            test_sample["target"] = target
            
        # Load the textured mesh (GLB format)
        test_sample["texture_mesh"] = trimesh.load(os.path.join(base_path,"final.glb"))
        
        # Store the conditioning information (images or text)
        test_sample["cond"] = cond
        
        # Store an identifier for this sample
        test_sample["id"] = id_
        return test_sample
from scipy.spatial.transform import Rotation as R   

if __name__ == "__main__":
    
    import json
    import tqdm
    from transformers import CLIPProcessor, CLIPModel
    device = "cuda"  # Use GPU for faster computation
    
    # Load pretrained CLIP model and processor
    model = CLIPModel.from_pretrained("../pretrained-models/openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("../pretrained-models/openai/clip-vit-large-patch14")
    model.to(device)  # Move model to GPU
    
    sample_cache_dir = './cache'  # Directory to cache intermediate results
    # Create the cache directory if it does not exist
    if not os.path.exists(sample_cache_dir):
        os.makedirs(sample_cache_dir)
        

    

    # !!! NOTE: The `trellis_out_path` below is currently a placeholder.
    # To properly evaluate Chamfer Distance (CD) and F-score, replace it with
    # the actual trellis/target result mesh(in glb format).

    
    # Example 1: Multi-image conditional evaluation
    test_sample = get_test_sample(
        "./results/multi-image/1", 
        cond=[Image.open(f"./test_200_new/1/1_{j}.png") for j in [1,2,3]],  # List of input images
        trellis_out_path="./results/multi-image/1/final.glb",
        id_=1
    )
    
    # Evaluate the mesh with optional target and inner meshes
    res = eval_mesh(
        test_sample["surface"],
        input_condition=test_sample["cond"],
        target=test_sample["target"] if "target" in test_sample else None,
        inner=test_sample["inner"] if "inner" in test_sample else None,
        render=0,  # No rendering
        device=device,
        cache_dir=sample_cache_dir,
        base_id=test_sample["id"],
        render_mesh=test_sample["texture_mesh"]
    )
    print(res)
    
    # Example 2: Single-image conditional evaluation
    test_sample = get_test_sample(
        "./results/single-image/1", 
        cond=[Image.open(f"./test_200_new/1/1_{3}.png")],  # Single input image
        trellis_out_path="./results/single-image/1/final.glb",
        id_=1
    )
    
    res = eval_mesh(
        test_sample["surface"],
        input_condition=test_sample["cond"],
        target=test_sample["target"] if "target" in test_sample else None,
        inner=test_sample["inner"] if "inner" in test_sample else None,
        render=0,
        device=device,
        cache_dir=sample_cache_dir,
        base_id=test_sample["id"],
        render_mesh=test_sample["texture_mesh"]
    )
    print(res)
    
    # Example 3: Text-conditioned evaluation
    test_sample = get_test_sample(
        "./results/text/1", 
        cond="A big apple.",  # Text prompt as condition
        trellis_out_path="./results/text/1/final.glb",
        id_=1
    )
    
    res = eval_mesh(
        test_sample["surface"],
        input_condition=test_sample["cond"],
        target=test_sample["target"] if "target" in test_sample else None,
        inner=test_sample["inner"] if "inner" in test_sample else None,
        render=0,
        device=device,
        cache_dir=sample_cache_dir,
        base_id=test_sample["id"],
        render_mesh=test_sample["texture_mesh"]
    )
    print(res)
