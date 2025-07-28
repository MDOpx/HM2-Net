import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap

def interactive_uncertainty_viewer(image_data):
    # 创建自定义colormap（浅黄到红）
    colors = [(1, 1, 0.7), (1, 0, 0)]  # 浅黄到红
    cmap_name = 'yellow_to_red'
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

    # # 生成示例图像数据（替换为你的实际数据）
    # image_data = np.random.randint(0, 256, (300, 300))

    # 创建图形和子图布局
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # 初始显示图像
    img = ax.imshow(image_data, cmap=custom_cmap, vmin=0, vmax=255)
    plt.colorbar(img, ax=ax, label='xiangsu')

    # 创建滑块轴
    ax_min = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_max = plt.axes([0.25, 0.1, 0.65, 0.03])

    # 创建滑块控件
    min_slider = Slider(ax_min, 'min', 0, 255, valinit=0, valstep=1)
    max_slider = Slider(ax_max, 'max', 0, 255, valinit=255, valstep=1)

    # 滑块更新函数
    def update(val):
        img.set_clim([min_slider.val, max_slider.val])
        fig.canvas.draw_idle()

    # 注册滑块事件
    min_slider.on_changed(update)
    max_slider.on_changed(update)

    ax.set_title('box')
    plt.show()


# def interactive_uncertainty_viewer(entropy_map, initial_vmin=0.2, initial_vmax=1.0):
#     # 初始 colormap：白 → 红
#     use_reversed = False
#     cmap_normal = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
#     cmap_reversed = LinearSegmentedColormap.from_list("red_white", ["red", "white"])
#     current_cmap = cmap_normal
#
#     print(f"📍 Entropy at (0, 0): {entropy_map[0, 0]:.4f}")
#     fig, ax = plt.subplots(figsize=(7, 6))
#     plt.subplots_adjust(bottom=0.35)
#
#     img = ax.imshow(entropy_map, cmap=current_cmap, vmin=initial_vmin, vmax=initial_vmax)
#     cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
#     ax.set_title(f"Uncertainty Map (vmin={initial_vmin:.2f}, vmax={initial_vmax:.2f})")
#     ax.axis('off')
#
#     # 滑块控件
#     ax_vmin = plt.axes([0.2, 0.2, 0.65, 0.03])
#     ax_vmax = plt.axes([0.2, 0.1, 0.65, 0.03])
#
#     slider_vmin = Slider(ax_vmin, 'vmin', 0.0, 1.0, valinit=initial_vmin, valstep=0.01)
#     slider_vmax = Slider(ax_vmax, 'vmax', 0.0, 1.0, valinit=initial_vmax, valstep=0.01)
#
#     def update(val):
#         vmin = slider_vmin.val
#         vmax = slider_vmax.val
#         if vmin >= vmax:
#             return
#         img.set_clim(vmin=vmin, vmax=vmax)
#         ax.set_title(f"Uncertainty Map (vmin={vmin:.2f}, vmax={vmax:.2f})")
#         fig.canvas.draw_idle()
#
#     slider_vmin.on_changed(update)
#     slider_vmax.on_changed(update)
#
#     # ✅ 按键响应：切换 colormap
#     def on_key(event):
#         nonlocal use_reversed, current_cmap
#         if event.key == 'r':
#             use_reversed = not use_reversed
#             current_cmap = cmap_reversed if use_reversed else cmap_normal
#             img.set_cmap(current_cmap)
#             print(f"🎨 Colormap switched to: {'red → white' if use_reversed else 'white → red'}")
#             fig.canvas.draw_idle()
#
#     fig.canvas.mpl_connect("key_press_event", on_key)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--entropy_path', type=str,
#                         default='/home/shuoxing/data/TransUNet/Results-Final-MFC/unmapdir_Kvasir/test/cju0s2a9ekvms080138tjjpxr.png',
#                         help='路径：entropy numpy 文件 (.npy)')
#     args = parser.parse_args()
#
#     entropy = cv2.imread(args.entropy_path)
#     entropy = np.clip(entropy, 0, 1)
#     # entropy = 1 - entropy
#     print(entropy.shape)
#     # 修复维度：如果是 (1, H, W)，转为 (H, W)
#     # if entropy.ndim == 3 and entropy.shape[0] == 1:
#     #     entropy = entropy[:,:,0]
#     if entropy.ndim == 3:
#         entropy = entropy[:,:,0]
#
#     interactive_uncertainty_viewer(entropy)

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def interactive_uncertainty_viewer(img):
    # 读取原始图像

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建图形界面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3)

    # 初始颜色设置 (BGR格式)
    color1 = np.array([255, 255, 200])  # 红色
    color2 = np.array([240, 0, 0])  # 蓝色

    # 创建伪彩图
    def create_pseudo_color(c1, c2):
        pseudo = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for i in range(3):  # 对每个颜色通道
            pseudo[..., i] = np.interp(gray, [0, 255], [c1[i], c2[i]])
        return pseudo

    # 初始显示
    pseudo_img = create_pseudo_color(color1, color2)
    im1 = ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    im2 = ax2.imshow(pseudo_img)
    ax1.set_title('Original Image')
    ax2.set_title('Pseudo Color')

    # 创建颜色滑块
    axcolor = 'lightgoldenrodyellow'
    ax_r1 = plt.axes([0.2, 0.2, 0.6, 0.02], facecolor=axcolor)
    ax_g1 = plt.axes([0.2, 0.17, 0.6, 0.02], facecolor=axcolor)
    ax_b1 = plt.axes([0.2, 0.14, 0.6, 0.02], facecolor=axcolor)
    ax_r2 = plt.axes([0.2, 0.1, 0.6, 0.02], facecolor=axcolor)
    ax_g2 = plt.axes([0.2, 0.07, 0.6, 0.02], facecolor=axcolor)
    ax_b2 = plt.axes([0.2, 0.04, 0.6, 0.02], facecolor=axcolor)

    sliders = [
        Slider(ax_r1, 'Color1 R', 0, 255, valinit=color1[0]),
        Slider(ax_g1, 'Color1 G', 0, 255, valinit=color1[1]),
        Slider(ax_b1, 'Color1 B', 0, 255, valinit=color1[2]),
        Slider(ax_r2, 'Color2 R', 0, 255, valinit=color2[0]),
        Slider(ax_g2, 'Color2 G', 0, 255, valinit=color2[1]),
        Slider(ax_b2, 'Color2 B', 0, 255, valinit=color2[2])
    ]

    # 更新函数
    def update(val):
        new_color1 = np.array([s.val for s in sliders[:3]])
        new_color2 = np.array([s.val for s in sliders[3:]])
        new_pseudo = create_pseudo_color(new_color1, new_color2)
        im2.set_data(new_pseudo)
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()

def change_color_single(image_data,output_name):
    # def create_pseudo_color(c1, c2):
    #     pseudo = np.zeros((*gray.shape, 3), dtype=np.uint8)
    #     for i in range(3):
    #         pseudo[..., i] = np.interp(gray, [0, 255], [c1[i], c2[i]])
    #     return pseudo

    #
    # # 生成伪彩图
    # gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    # pseudo_img = cv2.cvtColor(create_pseudo_color(color1_rgb, color2_rgb), cv2.COLOR_BGR2RGB)
    #
    # # 保存伪彩图
    # cv2.imwrite(output_name, cv2.cvtColor(pseudo_img, cv2.COLOR_RGB2BGR))


    gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # 创建图形界面
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.3)

    # 初始颜色设置 (BGR格式)
    color1 = np.array([255, 255, 200])  # 红色
    color2 = np.array([240, 0, 0])  # 蓝色

    # 转换颜色到RGB格式
    # color1_rgb = np.array([255, 255, 200])[::-1] / 255.0  # BGR转RGB并归一化
    # color2_rgb = np.array([240, 0, 0])[::-1] / 255.0  # BGR转RGB并归一化
    color1_rgb = np.array([255, 255, 200]) / 255.0  # BGR转RGB并归一化
    color2_rgb = np.array([240, 0, 0]) / 255.0  # BGR转RGB并归一化

    # 创建自定义colormap
    cmap = LinearSegmentedColormap.from_list('custom', [color1_rgb, color2_rgb])

    # 创建伪彩图
    def create_pseudo_color(c1, c2):
        pseudo = np.zeros((*gray.shape, 3), dtype=np.uint8)
        for i in range(3):  # 对每个颜色通道
            pseudo[..., i] = np.interp(gray, [0, 255], [c1[i], c2[i]])
        return pseudo

    # 初始显示
    pseudo_img = create_pseudo_color(color1, color2)
    pseudo_img = cv2.cvtColor(pseudo_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(output_name,pseudo_img)

    # 创建带colorbar的图像
    plt.figure(figsize=(8, 6))
    img = plt.imshow(gray, cmap=cmap)
    plt.colorbar(img, label='Intensity')
    plt.axis('off')

    # 保存带colorbar的图像
    colorbar_output = output_name.replace('.', '_pseudo.')
    plt.savefig(colorbar_output, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # # 显示伪彩图
    # im = ax1.imshow(pseudo_img)
    # ax1.set_title('Pseudo Color Image')
    # ax1.axis('off')
    # # 添加colorbar
    # cbar = fig.colorbar(im, ax=ax2, orientation='vertical')
    # cbar.set_label('Intensity Scale')
    # ax2.axis('off')
    #
    # # 保存带colorbar的图像
    # plt.savefig(output_name, bbox_inches='tight', dpi=300)
    # plt.close()
    # cv2.imwrite(output_name.replace('.', '_pseudo.'), pseudo_img)



def change_color(dataset,class_name):
    image_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/unmapdir_' + dataset + '/' + class_name
    output_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/recolorunmap_' + dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder = '/home/shuoxing/data/TransUNet/Results-Final-MFC/recolorunmap_' + dataset + '/' + class_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        img_name = os.path.join(image_folder, filename)
        output_name = os.path.join(output_folder, filename)
        if filename.endswith('.png'):
            print(filename)
            image_data = cv2.imread(
                img_name)  # 读取为BGR格式
            change_color_single(image_data,output_name)

if __name__ == '__main__':
    class_name = 'test'

    # dataset_name = 'CVC-300'
    # change_color(dataset_name,class_name)
    dataset_name = 'CVC-ClinicDB'
    change_color(dataset_name, class_name)
    # dataset_name = 'CVC-ColonDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'ETIS-LaribPolypDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'Kvasir'
    # change_color(dataset_name, class_name)
    # dataset_name = 'UDIAT'
    # change_color(dataset_name, class_name)
    # dataset_name = 'BUSI'
    # change_color(dataset_name, class_name)

    class_name = 'val'

    # dataset_name = 'CVC-300'
    # change_color(dataset_name,class_name)
    dataset_name = 'CVC-ClinicDB'
    change_color(dataset_name, class_name)
    # dataset_name = 'CVC-ColonDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'ETIS-LaribPolypDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'Kvasir'
    # change_color(dataset_name, class_name)
    # dataset_name = 'UDIAT'
    # change_color(dataset_name, class_name)
    # dataset_name = 'BUSI'
    # change_color(dataset_name, class_name)

    class_name = 'train'

    # dataset_name = 'CVC-300'
    # change_color(dataset_name,class_name)
    dataset_name = 'CVC-ClinicDB'
    change_color(dataset_name, class_name)
    # dataset_name = 'CVC-ColonDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'ETIS-LaribPolypDB'
    # change_color(dataset_name, class_name)
    # dataset_name = 'Kvasir'
    # change_color(dataset_name, class_name)
    # dataset_name = 'UDIAT'
    # change_color(dataset_name, class_name)
    # dataset_name = 'BUSI'
    # change_color(dataset_name, class_name)




