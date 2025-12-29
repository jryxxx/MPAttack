from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ntpath
import imageio
import numpy as np
import scipy.misc
import math
import cv2
# import chainer
import torch
import neural_renderer
from glob import glob
import os
import matplotlib.pyplot as plt
#############
### Utils ###
#############


def convert_as(src, trg):
    src = src.type_as(trg)
    if src.is_cuda:
        src = src.cuda(device=trg.get_device())
    return src


def get_params(carlaTcam, carlaTveh):  # carlaTcam: tuple of 2*3
    #scale = 0.39
    scale=0.328
    # scale = 0.38
    # calc eye
    eye = [0, 0, 0]
    for i in range(0, 3):
        # eye[i] = (carlaTcam[0][i] - carlaTveh[0][i]) * scale
        eye[i] = carlaTcam[0][i] * scale

    # calc camera_direction and camera_up
    pitch = math.radians(carlaTcam[1][0])
    yaw = math.radians(carlaTcam[1][1])
    roll = math.radians(carlaTcam[1][2])

    cam_direct = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
    cam_up = [math.cos(math.pi / 2 + pitch) * math.cos(yaw), math.cos(math.pi / 2 + pitch) * math.sin(yaw),
              math.sin(math.pi / 2 + pitch)]

    p_cam = eye
    p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
    p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
    p_l = [p_cam, p_dir, p_up]
    trans_p = []
    for p in p_l:
        if math.sqrt(p[0] ** 2 + p[1] ** 2) == 0:
            cosfi = 0
            sinfi = 0
        else:
            cosfi = p[0] / math.sqrt(p[0] ** 2 + p[1] ** 2)
            sinfi = p[1] / math.sqrt(p[0] ** 2 + p[1] ** 2)
        cossum = cosfi * math.cos(math.radians(carlaTveh[1][1])) + sinfi * math.sin(math.radians(carlaTveh[1][1]))
        sinsum = math.cos(math.radians(carlaTveh[1][1])) * sinfi - math.sin(math.radians(carlaTveh[1][1])) * cosfi
        trans_p.append([math.sqrt(p[0] ** 2 + p[1] ** 2) * cossum, math.sqrt(p[0] ** 2 + p[1] ** 2) * sinsum, p[2]])

    return trans_p[0], \
           [trans_p[1][0] - trans_p[0][0], trans_p[1][1] - trans_p[0][1], trans_p[1][2] - trans_p[0][2]], \
           [trans_p[2][0] - trans_p[0][0], trans_p[2][1] - trans_p[0][1], trans_p[2][2] - trans_p[0][2]]


########################################################################
############ Wrapper class for the chainer Neural Renderer #############
##### All functions must only use numpy arrays as inputs/outputs #######
########################################################################
class NMR(object):
    def __init__(self):
        # setup renderer
        renderer = neural_renderer.Renderer(camera_mode='look')
        self.renderer = renderer

    def to_gpu(self, device=0):

        self.cuda_device = device

    def forward_mask(self, vertices, faces):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
        Returns:
            masks: B X 256 X 256 numpy array
        '''
        self.faces = torch.autograd.Variable(faces.cuda())
        self.vertices = torch.autograd.Variable(vertices.cuda())

        self.masks = self.renderer.render_silhouettes(
            self.vertices, self.faces)

        masks = self.masks.data.get()
        return masks

    def forward_img(self, vertices, faces, textures):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        self.faces = faces
        self.vertices = vertices
        self.textures = textures
        self.images, _, _ = self.renderer.render(
            self.vertices, self.faces, self.textures)
        return self.images


########################################################################
################# Wrapper class a rendering PythonOp ###################
##### All functions must only use torch Tensors as inputs/outputs ######
########################################################################
class Render(torch.autograd.Function):
    # TODO(Shubham): Make sure the outputs/gradients are on the GPU
    def __init__(self, renderer):
        super(Render, self).__init__()
        self.renderer = renderer

    def forward(self, vertices, faces, textures=None):
        # B x N x 3
        # Flipping the y-axis here to make it align with the image coordinate system!
        vs = vertices
        vs[:, :, 1] *= -1
        fs = faces
        if textures is None:
            self.mask_only = True
            masks = self.renderer.forward_mask(vs, fs)
            return masks   # , convert_as(torch.Tensor(masks), vertices)
        else:
            self.mask_only = False
            ts = textures
            imgs = self.renderer.forward_img(vs, fs, ts)
            return imgs


########################################################################
############## Wrapper torch module for Neural Renderer ################
########################################################################
class NeuralRenderer(torch.nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """

    def __init__(self, img_size=720):
        super(NeuralRenderer, self).__init__()
        self.renderer = NMR()

        # rendering
        self.renderer.renderer.image_size = img_size

        # camera
        self.renderer.renderer.camera_mode = 'look'
        self.renderer.renderer.viewing_angle = 45
        # test example
        eye, camera_direction, camera_up = get_params(
            ((-25, 16, 20), (-45, 180, 0)), ((-45, 3, 0.8), (0, 0, 0)))
        self.renderer.renderer.eye = eye
        self.renderer.renderer.camera_direction = camera_direction
        self.renderer.renderer.camera_up = camera_up

        # light
        self.renderer.renderer.light_intensity_ambient = 0.5
        self.renderer.renderer.light_intensity_directional = 0.5
        self.renderer.renderer.light_color_ambient = [1, 1, 1]  # white
        self.renderer.renderer.light_color_directional = [1, 1, 1]  # white
        self.renderer.renderer.light_direction = [0, 0, 1]  # up-to-down

        self.renderer.to_gpu()

        self.proj_fn = None
        self.offset_z = 5.

        self.RenderFunc = Render(self.renderer)

    def ambient_light_only(self):
        # Make light only ambient.
        self.renderer.renderer.light_intensity_ambient = 1
        self.renderer.renderer.light_intensity_directional = 0

    def set_bgcolor(self, color):
        self.renderer.renderer.background_color = color

    def project_points(self, verts, cams):
        proj = self.proj_fn(verts, cams)
        return proj[:, :, :2]

    def forward(self, vertices, faces, textures=None):
        if textures is not None:
            return self.RenderFunc.forward(vertices, faces, textures)
        else:
            return self.RenderFunc.forward(vertices, faces)


def example():
    obj_file = '/media/bjh/disk-1.0TB/Paper/Paper2/model/test.obj'
    data_path = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/npz'
    mask_path = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/mask'
    img_save_dir = '/media/bjh/disk-1.0TB/Paper/Paper2/images'
    os.makedirs(img_save_dir, exist_ok=True)
    vertices, faces = neural_renderer.load_obj(obj_file)

    texture_mask = np.zeros((faces.shape[0], 2, 2, 2, 3), 'int8')
    with open('/media/bjh/disk-1.0TB/Paper/Paper2/model/faces_new.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)


    mask_renderer = NeuralRenderer(img_size=800)
    faces_var = torch.autograd.Variable(faces[None, :, :]).cuda(device=0)
    vertices_var = vertices[None, :, :].cuda(device=0)
    # Textures
    texture_size = 2
    textures = np.ones(
        (1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = torch.from_numpy(textures).cuda(device=0)
    textures = textures * texture_mask

    # data = np.load(data_path)
    data_lsit = glob(os.path.join(data_path, "*.npz"))
    for data_path in data_lsit:
        data = np.load(data_path)
        img = data['img']
        img = np.transpose(img, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        img = torch.from_numpy(img).cuda(device=0)
        veh_trans = data['veh_trans']
        cam_trans = data['cam_trans']
        print('*' * 50)
        print(veh_trans)
        print(cam_trans)        
        eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
        mask_renderer.renderer.renderer.eye = eye
        mask_renderer.renderer.renderer.camera_direction = camera_direction
        mask_renderer.renderer.renderer.camera_up = camera_up
        imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)

        filename = os.path.splitext(os.path.basename(data_path))[0]
        mask_file = os.path.join(mask_path, f"{filename}.png")
        mask = cv2.imread(mask_file)
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        mask = torch.from_numpy(mask.astype('float32')).cuda()

        total_img = (1 - mask) * img + (255 * imgs_pred) * mask
        total_img_tensor = total_img[0].permute(1, 2, 0) / 255.0
        image_numpy = total_img_tensor.detach().cpu().numpy()
        plt.imsave(f"{img_save_dir}/{filename}.png", image_numpy)

def example_1():
    obj_file = '/media/bjh/disk-1.0TB/Paper/Paper2/model/test.obj'
    data_path = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/npz'
    mask_path = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/mask'
    img_save_dir = '/media/bjh/disk-1.0TB/Paper/Paper2/images'
    os.makedirs(img_save_dir, exist_ok=True)
    vertices, faces = neural_renderer.load_obj(obj_file)

    texture_mask = np.zeros((faces.shape[0], 2, 2, 2, 3), 'int8')
    with open('/media/bjh/disk-1.0TB/Paper/Paper2/model/faces_new.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            texture_mask[int(face_id) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)


    mask_renderer = NeuralRenderer(img_size=800)
    faces_var = torch.autograd.Variable(faces[None, :, :]).cuda(device=0)
    vertices_var = vertices[None, :, :].cuda(device=0)
    # Textures
    texture_size = 2
    textures = np.ones(
        (1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
    textures = torch.from_numpy(textures).cuda(device=0)
    textures = textures * texture_mask

    data_path = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/npz/pos_17_az0_0_el-60_0_dist30_0.npz'
    mask_file = '/media/bjh/disk-1.0TB/CARLA/carla/dataset/mask/pos_17_az0_0_el-60_0_dist30_0.png'

    data = np.load(data_path)
    img = data['img']
    img = np.transpose(img, (2, 0, 1))
    img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    img = torch.from_numpy(img).cuda(device=0)
    veh_trans = data['veh_trans']
    cam_trans = data['cam_trans']
    print('*' * 50)
    print(veh_trans)
    print(cam_trans)        
    eye, camera_direction, camera_up = get_params(cam_trans, veh_trans)
    mask_renderer.renderer.renderer.eye = eye
    mask_renderer.renderer.renderer.camera_direction = camera_direction
    mask_renderer.renderer.renderer.camera_up = camera_up
    imgs_pred = mask_renderer.forward(vertices_var, faces_var, textures)

    mask = cv2.imread(mask_file)
    mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
    mask = torch.from_numpy(mask.astype('float32')).cuda()

    total_img = (1 - mask) * img + (255 * imgs_pred) * mask
    total_img_tensor = total_img[0].permute(1, 2, 0) / 255.0
    image_numpy = total_img_tensor.detach().cpu().numpy()
    plt.imsave(f"test.png", image_numpy)

if __name__ == '__main__':
    example()
