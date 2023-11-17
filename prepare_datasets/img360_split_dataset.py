import glob,random
import numpy as np
import shutil, os
import ipdb
import argparse

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    print("random seed set: {}".format(seed))

setup_seed(seed = 2023)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="replica or mp3d")
args = parser.parse_args()


DATASET = "replica" # "mp3d"

if DATASET == "mp3d":
    SCENE_SPLITS = {
        'train': ['sT4fr6TAbpF', 'E9uDoFAP3SH', 'VzqfbhrpDEA', 'kEZ7cmS4wCh', '29hnd4uzFmX', 'ac26ZMwG7aT',
                'i5noydFURQK', 's8pcmisQ38h', 'rPc6DW4iMge', 'EDJbREhghzL', 'mJXqzFtmKg4', 'B6ByNegPMKs',
                'JeFG25nYj2p', '82sE5b5pLXE', 'D7N2EKCX4Sj', '7y3sRwLe3Va', 'HxpKQynjfin', '5LpN3gDmAk7',
                'gTV8FGcVJC9', 'ur6pFq6Qu1A', 'qoiz87JEwZ2', 'PuKPg4mmafe', 'VLzqgDo317F', 'aayBHfsNo7d',
                'JmbYfDe2QKZ', 'XcA2TqTSSAj', '8WUmhLawc2A', 'sKLMLpTHeUy', 'r47D5H71a5s', 'Uxmj2M2itWa',
                'Pm6F8kyY3z2', 'p5wJjkQkbXX', '759xd9YjKW5', 'JF19kD82Mey', 'V2XKFyX4ASd', '1LXtFkjw3qL',
                '17DRP5sb8fy', '5q7pvUzZiYa', 'VVfe2KiqLaN', 'Vvot9Ly1tCj', 'ULsKaCPVFJR', 'D7G3Y4RVNrH',
                'uNb9QFRL6hY', 'ZMojNkEp431', '2n8kARJN3HM', 'vyrNrziPKCB', 'e9zR4mvMWw7', 'r1Q1Z4BcV1o',
                'PX4nDJXEHrG', 'YmJkqBEsHnH', 'b8cTxDM8gDG', 'GdvgFV5R1Z5', 'pRbA3pwrgk9', 'jh4fc5c5qoQ',
                '1pXnuDYAj8r', 'S9hNv5qa7GM', 'VFuaQ6m2Qom', 'cV4RVeZvu5T', 'SN83YJsR3w2'],
        'val': ['x8F5xyUWy9e', 'QUCTc6BB5sX', 'EU6Fwq7SyZv', '2azQ1b91cZZ', 'Z6MFQCViBuw', 'pLe4wQe7qrG',
                'oLBMNvg9in8', 'X7HyMhZNoso', 'zsNo4HB9uLZ', 'TbHJrupSAjP', '8194nk5LbLH'],
        'test': ['pa4otMbVnkk', 'yqstnuAEVhm', '5ZKStnWn8Zo', 'Vt2qJdWjCF2', 'wc2JMjhGNzB', 'WYY7iVyf5p8',
                'fzynW3qQPVF', 'UwV83HsGsw3', 'q9vSo1VnCiC', 'ARNzJeq3xxb', 'rqfALeAoiTq', 'gYvKGZ5eRqb',
                'YFuZgdQ5vWj', 'jtcxE69GiFV', 'gxdoqLR6rwA'],
    }
    SCENE_SPLITS['train_distractor'] = SCENE_SPLITS['train']
    SCENE_SPLITS['val_distractor'] = SCENE_SPLITS['val']
    SCENE_SPLITS['test_distractor'] = SCENE_SPLITS['test']
    
elif DATASET == "replica":
    
    SCENE_SPLITS = {}
    SCENE_SPLITS['train_distractor'] = ["apartment_0", "frl_apartment_0","frl_apartment_1","frl_apartment_2","frl_apartment_3","hotel_0","office_0","office_1","office_2","room_0"]#all_path[0:train_num] #random.choices(all_path, k=train_num)
    SCENE_SPLITS['val_distractor'] = ["apartment_1", "frl_apartment_4", "office_3", "room_1"]#all_path[train_num:train_num+val_num] #random.choices(all_path, k=val_num)
    SCENE_SPLITS['test_distractor'] = ["apartment_2", "frl_apartment_5", "office_4", "room_2"]#all_path[train_num+val_num: total_room_num] #random.choices(all_path, k=test_num)
    print("len train{}, len val{}, len test{}".format(len(SCENE_SPLITS['train_distractor']), len(SCENE_SPLITS['val_distractor']), len(SCENE_SPLITS['test_distractor'])))
    

for letter in ["A","B","C"]:
    current_dir ="./dataset_realEquirec_{}/train_{}/".format(DATASET,letter)
    all_path = glob.glob(current_dir + "*")
    all_rooms = [i.split(current_dir)[1] for i in all_path]
    print(len(all_rooms))
    print((all_rooms))


    train_rooms = list(set(all_rooms).intersection(set(SCENE_SPLITS['train_distractor'])))
    val_rooms = list(set(all_rooms).intersection(set(SCENE_SPLITS['val_distractor'])))
    test_rooms = list(set(all_rooms).intersection(set(SCENE_SPLITS['test_distractor'])))
    print("len train{}, len val{}, len test{}".format(len(train_rooms), len(val_rooms), len(test_rooms)))

    #ipdb.set_trace()
    
    print(val_rooms)

    COPY = True
    if COPY == True:
        for path in train_rooms:
            path = os.path.join(current_dir, path)
            for i,file in enumerate(sorted(os.listdir(path))):
                # absolute path
                file_path = os.path.join(path,file)
                src_path = file_path
                dst_path = file_path.replace("dataset_realEquirec_{}".format(DATASET),'dataset_realEquirec_{}_organized'.format(DATASET))
                
                dst_path_to_create_list = dst_path.split('/')[:-1]
                dst_path_to_create = '/'.join(dst_path_to_create_list)
                if not os.path.exists(dst_path_to_create):
                    os.makedirs(dst_path_to_create)
                    
                print(src_path)
                print(dst_path)
                #shutil.move(src_path, dst_path)
                shutil.copy(src_path, dst_path)

        for path in val_rooms:
            path = os.path.join(current_dir, path)
            for i,file in enumerate(sorted(os.listdir(path))):
                # absolute path
                file_path = os.path.join(path,file)
                src_path = file_path
                dst_path = file_path.replace("dataset_realEquirec_{}".format(DATASET),'dataset_realEquirec_{}_organized'.format(DATASET)).replace("train",'val')
                
                dst_path_to_create_list = dst_path.split('/')[:-1]
                dst_path_to_create = '/'.join(dst_path_to_create_list)
                if not os.path.exists(dst_path_to_create):
                    os.makedirs(dst_path_to_create)
                    
                print(src_path)
                print(dst_path)
                #shutil.move(src_path, dst_path)
                shutil.copy(src_path, dst_path)

        for path in test_rooms:
            path = os.path.join(current_dir, path)
            for i,file in enumerate(sorted(os.listdir(path))):
                # absolute path
                file_path = os.path.join(path,file)
                src_path = file_path
                dst_path = file_path.replace("dataset_realEquirec_{}".format(DATASET),'dataset_realEquirec_{}_organized'.format(DATASET)).replace("train",'test')
                
                dst_path_to_create_list = dst_path.split('/')[:-1]
                dst_path_to_create = '/'.join(dst_path_to_create_list)
                if not os.path.exists(dst_path_to_create):
                    os.makedirs(dst_path_to_create)
                    
                print(src_path)
                print(dst_path)
                #shutil.move(src_path, dst_path)
                shutil.copy(src_path, dst_path)