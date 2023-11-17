import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sound_spaces_data_path", help="path to downloaded soundspace data folder")
parser.add_argument("--dataset", help="replica or mp3d")
args = parser.parse_args()

def get_echocentric_wav_paths(ROOT_PATH,room_name):

    def copy_file_for_original_image2reverb(path_tmp,room_name,degree,file_name):
        image2reverb_dataset_path = "./dataset_realEquirec_{}/train_B".format(DATASET)
        src = path_tmp
        loc = file_name.split('_')[0]
        tmp = os.path.join(image2reverb_dataset_path,room_name)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        dst = os.path.join(tmp,loc+'_'+degree+'.wav')
        #if os.path.isfile(dst):
        shutil.copyfile(src, dst)
        print('copy from {} to {}'.format(src,dst))
        #else:
        #    print("{} already existed".format(dst))

    echocentric_wav_paths_dict = {}
    
    # for room_name in os.listdir(ROOT_PATH):
        
    with tqdm(total=len(os.listdir(os.path.join(ROOT_PATH,room_name)))) as pbar:
        pbar.set_description('Processing echocentric_wav:'+ room_name)
        for degree in os.listdir(os.path.join(ROOT_PATH,room_name)):
            # echocentric_wav_paths_list = []
            for file_name in os.listdir(os.path.join(ROOT_PATH,room_name,degree)):
                emit_pos = file_name.split("_")[0]
                receiver_pos = file_name.split("_")[1].split(".")[0]
                if emit_pos == receiver_pos:
                    # echocentric_wav_paths_list.append(os.path.join(ROOT_PATH,room_name,degree,file_name))
                    path_tmp = os.path.join(ROOT_PATH,room_name,degree,file_name)

                    copy_file_for_original_image2reverb(path_tmp,room_name,degree,file_name)

                    #magnitude_imp = self._compute_spect(path_tmp)
                    # fs_imp, sig_imp = wavfile.read(path_tmp)
                    #echocentric_wav_paths_dict[room_name+"_"+emit_pos+"_"+degree] = magnitude_imp
            pbar.update()
                        
                # echocentric_wav_paths_dict[room_name+"_"+emit_pos+"_"+degree] = echocentric_wav_paths_list
    return echocentric_wav_paths_dict

def get_views_dict(ROOT_OBS_PATH = "./img_mp3d",room_name = None):

    def copy_file_for_original_image2reverb_RGB(rgb_path,room_name,pos,degree):
        image2reverb_dataset_path = "./dataset_realEquirec_{}/train_A".format(DATASET)
        tmp = os.path.join(image2reverb_dataset_path,room_name)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        save_path = os.path.join(tmp,pos+'_'+degree+'.jpg')
        src = rgb_path
        dst = save_path
        shutil.copyfile(src, dst)
        print('copy from {} to {}'.format(src,dst))
    
    def copy_file_for_original_image2reverb_depth(depth_path,room_name,pos,degree):
        image2reverb_dataset_path = "./dataset_realEquirec_{}/train_C".format(DATASET)   
        tmp = os.path.join(image2reverb_dataset_path,room_name)
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        #im = Image.fromarray(depth_array)
        save_path = os.path.join(tmp,pos+'_'+degree+'.npy')
        src = depth_path
        dst = save_path
        shutil.copyfile(src, dst)
        print('saved depth array to {}'.format(save_path))
    
    view_dict = {}
    
    TMP_ROOM_DIR = os.path.join(ROOT_OBS_PATH,room_name)
    RGB_TMP_DIR = os.path.join(TMP_ROOM_DIR,"rgb")
    Depth_TMP_DIR = os.path.join(TMP_ROOM_DIR,"depth")
    file_names = os.listdir(RGB_TMP_DIR)
    with tqdm(total=len(file_names)) as pbar1:
        for name in file_names:
            key = name.split(".")[0].split("_")
            pos = key[0]
            angle = key[1]
            #view_dict[room_name+"_"+str(pos)+"_"+str(angle)] = data[key]
            rgb_path = os.path.join(RGB_TMP_DIR,name)
            depth_path = os.path.join(Depth_TMP_DIR,name.replace("jpg","npy"))
            copy_file_for_original_image2reverb_RGB(rgb_path  ,room_name,str(pos),str(angle))
            copy_file_for_original_image2reverb_depth(depth_path  ,room_name,str(pos),str(angle))
            pbar1.update()
            
    return view_dict

def get_commen_wav_view_dict(echocentric_wav_paths_dict,view_dict):
    for key in view_dict.keys():
        for sub_key in view_dict[key].keys():
            if sub_key in echocentric_wav_paths_dict[key].keys():
                view_dict[key][sub_key]["rir_wav"] = echocentric_wav_paths_dict[key][sub_key]
    return view_dict

if __name__ == "__main__":
    DATASET =  args.dataset #"replica" #"mp3d"
    ROOT_PATH_RIR = args.sound_spaces_data_path + "binaural_rirs/" + DATASET
    rooms_with_rir = os.listdir(ROOT_PATH_RIR)
    print(len(rooms_with_rir))

    ROOT_PATH_PANO = "./img_{}".format(DATASET) 
    rooms_with_pano = os.listdir(ROOT_PATH_PANO)
    print(len(rooms_with_pano))

    union = list(set(rooms_with_rir).intersection(set(rooms_with_pano)))
    print(len(union))
    
    
    for commun_room in union:
        get_echocentric_wav_paths(ROOT_PATH_RIR,commun_room)
        get_views_dict(ROOT_OBS_PATH = ROOT_PATH_PANO,room_name = commun_room)
    
    
    