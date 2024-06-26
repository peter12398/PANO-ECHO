##########
#script modified from https://github.com/HS-YN/DAPS/tree/b2823fb7611f5aa2f0c4ca9c4705570bfa965eb8/DAPS
##########
import os
import argparse
import numpy as np

import pickle

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_angle_axis, quat_from_coeffs
from tqdm import tqdm
from utils import load_metadata
import ipdb, shutil

parser = argparse.ArgumentParser()
parser.add_argument("--sound_spaces_data_path", help="path to downloaded soundspace data folder")
parser.add_argument("--dataset", help="replica or mp3d")
args = parser.parse_args()

print(args)

rgb_sensor = True  # rgb image
depth_sensor = True  # depth image
semantic_sensor = False  # semantic, semantic_cat image

"""
with open('scene_pos_coord_2d.pkl', 'rb') as g:
    scene_position_coord = pickle.load(g)
scene_position_coord = scene_position_coord[0]
"""
#scenes = list(scene_position_coord.keys())


def get_positions(dataset = "mp3d", scene = None, metadata_dir = None): #mp3d or replica
    positions = []
    scene_obs = dict()
    scene_metadata_dir = os.path.join(metadata_dir, scene)
    points, graph = load_metadata(scene_metadata_dir)
    for node in graph.nodes():
        agent_position = graph.nodes()[node]['point']
        print("agent_position",agent_position)
        #ipdb.set_trace()
        angles = [0, 90, 180, 270]
        positions.append(tuple((node, agent_position)))
    return positions
            

DATASET = args.dataset #"replica" #"mp3d"

save_path = "."
sound_spaces_data_path = args.sound_spaces_data_path
scenes_already_generated = save_path, 'img_{}'.format(DATASET) 
metadata_dir = sound_spaces_data_path + '/metadata/' + DATASET
    
for scene in os.listdir(metadata_dir):
    if not scene in scenes_already_generated:
        print("generating data for new scene :{}".format(scene))
    
        save_img_path = os.path.join(save_path, 'img_{}'.format(DATASET), scene)
        save_img_path_depth = os.path.join(save_img_path, 'depth')
        save_img_path_rgb = os.path.join(save_img_path, 'rgb')
        save_img_path_semantic = os.path.join(save_img_path, 'semantic')
        save_img_path_semantic_cat = os.path.join(save_img_path, 'semantic_cat')

        if os.path.exists(save_img_path):
            shutil.rmtree(save_img_path)
        os.makedirs(save_img_path_rgb)
        os.makedirs(save_img_path_depth)
        #os.makedirs(save_img_path_semantic)
        #os.makedirs(save_img_path_semantic_cat)
        
        if DATASET == 'replica':
            scene_path = os.path.join(sound_spaces_data_path, "scene_datasets", "replica", scene, 'habitat/mesh_semantic.ply')
        else:
            scene_path = os.path.join(sound_spaces_data_path, "scene_datasets","mp3d/{:s}/{:s}.glb".format(scene, scene))
            #scene_mesh_dir = os.path.join('data/scene_datasets', dataset, scene, scene + '.glb')
        #scene_config = os.path.join(sound_spaces_data_path, "scene_datasets","/{}/{}.scene_dataset_config.json".format(DATASET,DATASET))
        scene_config = os.path.join(sound_spaces_data_path, "/scene_datasets/mp3d/mp3d.scene_dataset_config.json")
        h = 1.5
        width = 1024 #512 #384
        height = 512 #256 #192
        sim_settings = {
            "width": width,  # Spatial resolution of the observations
            "height": height,
            "scene": scene_path,  # Scene path
            "scene_dataset": scene_config,  # the scene dataset configuration files
            "default_agent": 0,
            "sensor_height": h,  # Height of sensors in meters
            "color_sensor": rgb_sensor,  # RGB sensor
            "depth_sensor": depth_sensor,  # Depth sensor
            "semantic_sensor": semantic_sensor,  # Semantic sensor
            "seed": 1,  # used in the random navigation
            "enable_physics": False,  # kinematics only
        }
        threshold = 0.01 * width * height

        def make_cfg(settings):
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.gpu_device_id = 0
            sim_cfg.scene_id = settings["scene"]
            sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
            sim_cfg.enable_physics = settings["enable_physics"]

            # Note: all sensors must have the same resolution
            sensor_specs = []

            color_sensor_spec = habitat_sim.EquirectangularSensorSpec()
            color_sensor_spec.uuid = "color_sensor"
            color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            color_sensor_spec.resolution = [settings["height"], settings["width"]]
            color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
            sensor_specs.append(color_sensor_spec)

            depth_sensor_spec = habitat_sim.EquirectangularSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [settings["height"], settings["width"]]
            depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
            sensor_specs.append(depth_sensor_spec)

            semantic_sensor_spec = habitat_sim.EquirectangularSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
            semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
            semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR
            sensor_specs.append(semantic_sensor_spec)

            # Here you can specify the amount of displacement in a forward action and the turn angle
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = sensor_specs
            agent_cfg.action_space = {
                "move_forward": habitat_sim.agent.ActionSpec(
                    "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
                ),
                "turn_left": habitat_sim.agent.ActionSpec(
                    "turn_left", habitat_sim.agent.ActuationSpec(amount=90.0)
                ),
                "turn_right": habitat_sim.agent.ActionSpec(
                    "turn_right", habitat_sim.agent.ActuationSpec(amount=90.0)
                ),
                "look_left": habitat_sim.agent.ActionSpec(
                    "look_left", habitat_sim.agent.ActuationSpec(amount=90.0)
                ),
                "look_up": habitat_sim.agent.ActionSpec(
                    "look_up", habitat_sim.agent.ActuationSpec(amount=90.0)
                ),
                "look_down": habitat_sim.agent.ActionSpec(
                    "look_down", habitat_sim.agent.ActuationSpec(amount=90.0)
                ),
            }

            return habitat_sim.Configuration(sim_cfg, [agent_cfg])

        cfg = make_cfg(sim_settings)
        sim = habitat_sim.Simulator(cfg)
        # semantic_cat image settings
        scene_sem = sim.semantic_scene

        # move around each positions in scene and capture image
        #positions_orig = scene_position_coord[scene] # len(scene_position_coord[scene].keys()) = 489
        positions = get_positions(dataset = DATASET, scene = scene, metadata_dir=metadata_dir) #len(positions) = 20707
        #ipdb.set_trace()
        angles = [0,90,180,270]
        for (node,pos) in positions:
            #coord = positions[pos]
            coord = pos
            current = coord #[coord[0], coord[2]-h, -coord[1]]

            for angle in angles:
                agent_rotation = quat_to_coeffs(quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))).tolist()
                        
                # Set agent state
                agent = sim.initialize_agent(sim_settings["default_agent"])
                agent_state = habitat_sim.AgentState()
                agent_state.position = np.array(current)  # in world space
                agent_state.rotation = np.array(agent_rotation)  # in world space
                agent.set_state(agent_state)

                # Get agent state
                agent_state = agent.get_state()
                print(node, "agent_state: position", agent_state.position, "rotation", agent_state.rotation)

                action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

                observations = sim.get_sensor_observations()

                # rgb image
                rgb_obs = observations["color_sensor"]
                rgb_img = Image.fromarray(rgb_obs[:,:,:3])
                rgb_img.save(os.path.join(save_img_path_rgb, '{:d}_{:d}.jpg'.format(node,angle)))

                # depth image
                depth_obs = observations["depth_sensor"]
                #ipdb.set_trace()
                np.save(os.path.join(save_img_path_depth, '{:d}_{:d}.npy'.format(node,angle)), depth_obs)
                #depth_img = Image.fromarray((depth_obs / 10 * 65536).astype(np.uint16))
                #depth_img.save(os.path.join(save_img_path_depth, '{:d}_{:d}.png'.format(node,angle)))

                """
                semantic_obs = observations["semantic_sensor"]
                # semantic image
                semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
                semantic_img.putpalette(d3_40_colors_rgb.flatten())
                semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
                semantic_img = semantic_img.convert("RGB")
                semantic_img.save(os.path.join(save_img_path_semantic, '{:d}.jpg'.format(pos)))
                # semantic_cat_image
                sem_unique, sem_idx = np.unique(semantic_obs, return_inverse=True)
                new_cat = sem_unique.copy()
                for idx, uni in enumerate(sem_unique):
                    new_cat[idx] = instance_id_to_name[uni]
                semantic_obs_cat = new_cat[sem_idx].reshape(semantic_obs.shape).astype('uint8')

                # wrapper for compact labeling
                for i in range(semantic_obs_cat.shape[0]):
                    for j in range(semantic_obs_cat.shape[1]):
                        if np.count_nonzero(semantic_obs_cat == semantic_obs_cat[i][j]) < threshold:
                            semantic_obs_cat[i][j] = 0
                        if semantic_obs_cat[i][j] in MP_TO_COMP:
                            semantic_obs_cat[i][j] = MP_TO_COMP[semantic_obs_cat[i][j]]

                semantic_img_cat = Image.fromarray(semantic_obs_cat)
                semantic_img_cat.save(os.path.join(save_img_path_semantic_cat, '{:d}.png'.format(pos)))
                """
        sim.close()
    else:
        print("Already generated data for scene :{}".format(scene))

