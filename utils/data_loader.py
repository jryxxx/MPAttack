import neural_renderer
import utils.nr_utils as nmr
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import mmcv
import math
from mmdet.structures import DetDataSample


class ApplyTexture:
    def __init__(self, data_dir, texts, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        self.data_dir = data_dir
        self.img_size = img_size
        textures = np.ones(
            (1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures_adv = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = faces[None, :, :]
        self.vertices_var = vertices[None, :, :]
        self.mask_renderer = nmr.NeuralRenderer(img_size=img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        self.texts = texts

    def set_textures(self, textures_adv):
        self.textures_adv = textures_adv

    def apply(self, file_name):
        file = os.path.join(self.data_dir, file_name)
        data = np.load(file, allow_pickle=True)
        veh_trans, cam_trans = data['veh_trans'], data['cam_trans']

        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up

        imgs_pred = self.mask_renderer.forward(
            self.vertices_var, self.faces_var, self.textures_adv)

        # mmcv imread
        # img = mmcv.imread(img_file)
        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        # numpy load
        img = data['img']
        img = img[:, :, ::-1]
        img_clean = img.copy()

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)

        pred_np = imgs_pred.detach().cpu().numpy()
        pred_np = pred_np[0].transpose(1, 2, 0)  # CHW -> HWC
        pred_np = (pred_np * 255).clip(0,
                                       255).astype(np.uint8)  # 归一化到 [0, 255]
        imgs_pred = imgs_pred / torch.max(imgs_pred)
        cv2.imwrite("/root/ovd-attack/test/test_pred.png", pred_np)

        if self.ret_mask:
            mask_name = file_name.split('.')[0] + '.png'
            mask_file = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
            mask = torch.from_numpy(mask.astype('float32')).cuda()
            total_img = (1 - mask) * img + (255 * imgs_pred) * mask
            # convert to mmdetection tensor
            data_sample = DetDataSample()
            img_meta = {
                'img_shape': (800, 800),
                'ori_shape': (800, 800),
                'scale_factor': (1.0, 1.0),
                'texts': self.texts
            }
            data_sample.set_metainfo(img_meta)
            data_sample.text = self.texts
            data_dict = {'inputs': [total_img.squeeze(0)],
                         'data_samples': [data_sample]}
            return {
                'img': data_dict,
                'img_clean': img_clean,
                'pred': imgs_pred,
                'mask': mask.unsqueeze(0),
                'filename': file_name
            }

        total_img = img + 255 * imgs_pred
        return total_img.squeeze(0), imgs_pred.squeeze(0)


class DatasetAdv(Dataset):
    def __init__(self, data_dir, texts, img_size, texture_size, faces, vertices, distence=None, mask_dir=None, ret_mask=False):
        self.data_dir = data_dir
        self.files = []
        files = os.listdir(data_dir)
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                data = np.load(os.path.join(self.data_dir, file))
                veh_trans = data['veh_trans']
                cam_trans = data['cam_trans']
                dis = (cam_trans - veh_trans)[0, :]
                dis = np.sum(dis ** 2)
                if dis <= distence:
                    self.files.append(file)
        self.img_size = img_size
        textures = np.ones(
            (1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures_adv = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = faces[None, :, :]
        self.vertices_var = vertices[None, :, :]
        self.mask_renderer = nmr.NeuralRenderer(img_size=img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        self.texts = texts

    def set_textures(self, textures_adv):
        self.textures_adv = textures_adv

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        data = np.load(file, allow_pickle=True)
        veh_trans, cam_trans = data['veh_trans'], data['cam_trans']

        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        self.mask_renderer.renderer.renderer.eye = eye
        self.mask_renderer.renderer.renderer.camera_direction = camera_direction
        self.mask_renderer.renderer.renderer.camera_up = camera_up

        imgs_pred = self.mask_renderer.forward(
            self.vertices_var, self.faces_var, self.textures_adv)

        # mmcv imread
        # img = mmcv.imread(img_file)
        # img = mmcv.imconvert(img, 'bgr', 'rgb')
        # numpy load
        img = data['img']
        img = img[:, :, ::-1]
        img_clean = img.copy()

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)

        imgs_pred = imgs_pred / torch.max(imgs_pred)

        if self.ret_mask:
            mask_file = os.path.join(
                self.mask_dir, "%s.png" % self.files[index][:-4])
            mask = cv2.imread(mask_file)
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
            mask = torch.from_numpy(mask.astype('float32')).cuda()
            total_img = (1 - mask) * img + (255 * imgs_pred) * mask
            clean_img = img * mask
            # convert to mmdetection tensor
            data_sample = DetDataSample()
            img_meta = {
                'img_shape': (800, 800),
                'ori_shape': (800, 800),
                'scale_factor': (1.0, 1.0)
            }
            data_sample.set_metainfo(img_meta)
            data_sample.text = self.texts
            data_dict = {'inputs': [total_img.squeeze(0)],
                         'data_samples': [data_sample]}
            return {
                'index': index,
                'data': data_dict,
                'img_clean': img_clean,
                'pred': imgs_pred * 255.0,
                'clean': clean_img,  # for cal mse
                'mask': mask.unsqueeze(0),
                'filename': self.files[index],
            }

        total_img = img + 255 * imgs_pred
        return index, total_img.squeeze(0), imgs_pred.squeeze(0), self.files[index]

    def __len__(self):
        return len(self.files)


def load_npz(file_path):
    """
    Load a .npz file and return the data as a dictionary.
    :param file_path: Path to the .npz file
    :return: Dictionary containing the data
    """
    data = np.load(file_path, allow_pickle=True)
    return {key: data[key] for key in data.files} if isinstance(data, np.lib.npyio.NpzFile) else data


def check_data_shape(data_dir):
    """
    Check the shape of the data in the specified directory.
    :param data_dir: Directory containing the .npz files
    """
    shape_list = []
    for file in os.listdir(data_dir):
        data = load_npz(os.path.join(data_dir, file))
        shape = data['img'].shape
        shape_list.append(shape)
    print(set(shape_list))


def convert_npz_to_png(data_dir, save_dir):
    """
    Convert .npz files to .png images and save them in the specified directory.
    :param data_dir: Directory containing the .npz files
    :param save_dir: Directory to save the converted .png images
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(data_dir):
        data = load_npz(os.path.join(data_dir, file))
        img = data['img']
        cv2.imwrite(os.path.join(save_dir, file[:-4] + '.png'), img)


def test_data_loader():
    obj_file = 'data/audi_et_te.obj'
    vertices, faces, textures = neural_renderer.load_obj(
        filename_obj=obj_file, texture_size=6, load_texture=True)
    # Camouflage Textures
    faces_file = '/root/ovd-attack/data/exterior_face.txt'
    textures_file = '/root/ovd-attack/textures/texture_camouflage.npy'
    texture_content_adv = torch.from_numpy(
        np.load(textures_file)).cuda(device=0)
    texture_origin = textures[None, :, :, :, :, :].cuda(device=0)
    texture_mask = np.zeros((faces.shape[0], 6, 6, 6, 3), 'int8')
    with open(faces_file, 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            if face_id != '\n':
                texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)

    def cal_texture(texture_content):
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
        return texture_origin * (1 - texture_mask) + texture_mask * textures
    mask_dir = '/root/autodl-tmp/data/masks/'
    dataset = DatasetAdv(
        '/root/autodl-tmp/data/phy_attack/train/', "car . ", 800, 6, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)

    def collate_fn(batch):
        return batch
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    textures_adv = cal_texture(texture_content_adv)
    dataset.set_textures(textures_adv)

    for sample in loader:
        print(sample[0]['img']['inputs'][0].shape)
        print(sample[0]['filename'])
        img = sample[0]['img']['inputs'][0]
        tmp_img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        cv2.imwrite(f"tmp.png", np.clip(
            tmp_img, 0, 255).astype(np.uint8)[:, :, ::-1])
        break


def test_input_data():
    # mmdetection input
    # image -> opencv/mmcv(RBG) -> torch(permute((2, 0, 1)))
    image_file = '/root/autodl-tmp/data/images/train/data0.png'
    data = cv2.imread(image_file)
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    data = np.transpose(data, (2, 0, 1))
    data_tensor = torch.from_numpy(data)
    print(data_tensor)
    print("*" * 20)
    data = mmcv.imread(image_file)
    data = mmcv.imconvert(data, 'bgr', 'rgb')
    data_tensor = torch.from_numpy(
        data).permute((2, 0, 1)).cuda()
    print(data_tensor)
    print("*" * 20)
    data = load_npz('/root/autodl-tmp/data/phy_attack/train/data0.npz')
    data = data['img'][:, :, ::-1]
    print(data)
