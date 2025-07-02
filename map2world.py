import numpy as np
from PIL import Image
import yaml
import sys
import os

def generate_model(map_folder, map, model_folder):
    # 读取.yaml文件，获取分辨率和原点
    yaml_path = os.path.join(map_folder, map + '.yaml')
    with open(yaml_path, 'r') as f:
        map_yaml = yaml.safe_load(f)
    resolution = map_yaml['resolution']
    origin_x, origin_y, _ = map_yaml['origin']
    
    # 读取.pgm文件
    pgm_path = os.path.join(map_folder, map + '.pgm')
    img = Image.open(pgm_path).convert('L')
    data = np.array(img)
    

    model_name = map
    # 自动创建输出文件夹
    os.makedirs(model_folder, exist_ok=True)

    # 生成SDF模型
    sdf_file_path = os.path.join(model_folder, 'model.sdf')
    sdf_template = '''
    <?xml version="1.0"?>
        <sdf version="1.8">
            <model name="{model_name}">
                <pose>0 0 0 0 0 0</pose>
                <static>true</static> 
                {{blocks}}
            </model>
        </sdf>'''.format(model_name=model_name)
    block_template = '''
        <link name="obstacle_{id}">
            <pose>{x} {y} 0 0 0 0</pose>
            <static>true</static>
            <collision name="collision">
                <geometry><box><size>{size_x} {size_y} 1.0</size></box></geometry>
            </collision>
            <visual name="visual">
                <geometry><box><size>{size_x} {size_y} 1.0</size></box></geometry>
                <material>
                    <ambient>0.1 0.1 0.1 1</ambient>
                    <diffuse>0 0.01 0.05 1</diffuse>
                    <specular>0 0.01 0.05 1</specular>
                </material>
            </visual>
        </link>'''
    # 消除孤立像素点（简单消噪）
    def remove_isolated_points(data):
        h, w = data.shape
        new_data = data.copy()
        for y in range(1, h-1):
            for x in range(1, w-1):
                if data[y, x] < 50:
                    # 检查8邻域
                    neighbors = data[y-1:y+2, x-1:x+2]
                    if np.sum(neighbors < 50) <= 1:
                        new_data[y, x] = 255  # 设为非障碍物
        return new_data

    data = remove_isolated_points(data)
    # 合并相邻障碍物像素为最大矩形块
    def merge_rectangles(data):
        h, w = data.shape
        visited = np.zeros_like(data, dtype=bool)
        rectangles = []
        for y in range(h):
            for x in range(w):
                if data[y, x] < 50 and not visited[y, x]:
                    # 找到一个新块，扩展为最大矩形
                    max_w = 1
                    while x + max_w < w and data[y, x + max_w] < 50 and not visited[y, x + max_w]:
                        max_w += 1
                    max_h = 1
                    expand = True
                    while expand and y + max_h < h:
                        for dx in range(max_w):
                            if data[y + max_h, x + dx] >= 50 or visited[y + max_h, x + dx]:
                                expand = False
                                break
                        if expand:
                            max_h += 1
                    # 标记已访问
                    for dy in range(max_h):
                        for dx in range(max_w):
                            visited[y + dy, x + dx] = True
                    rectangles.append((y, x, max_h, max_w))
        return rectangles

    rectangles = merge_rectangles(data)
    blocks = []
    for i, (y, x, height, width) in enumerate(rectangles):
        x_pos = x * resolution + origin_x + (width * resolution) / 2.0
        y_pos = (data.shape[0] - y - height) * resolution + origin_y + (height * resolution) / 2.0
        blocks.append(block_template.format(
            id=i, size_x=width * resolution, size_y=height * resolution, x=x_pos, y=y_pos
        ))
    sdf_content = sdf_template.format(blocks="".join(blocks))
    with open(sdf_file_path, "w") as f:
        f.write(sdf_content)
    print(f"SDF map saved to {sdf_file_path}")
    
    
    # 自动生成model.config文件
    config_file_path = os.path.join(model_folder, "model.config")
    config_content = f'''<model>\n  <name>{model_name}</name>\n  <version>1.0</version>\n  <sdf version=\"1.5\">model.sdf</sdf>\n  <author><name>AutoGen</name><email>none@example.com</email></author>\n  <description>Auto generated model config</description>\n</model>'''
    with open(config_file_path, "w") as f:
        f.write(config_content)
    print(f"model.config saved to {config_file_path}")
    print(f"障碍物数量: {len(blocks)}，合并后实体数量")

def generate_world(model_folder, world_folder, world_name="default"):
    # 读取model.sdf内容
    sdf_path = os.path.join(model_folder, "model.sdf")
    with open(sdf_path, "r") as f:
        model_sdf_content = f.read()
    # 去除头部和<sdf>标签，只保留<model>...</model>
    start = model_sdf_content.find('<model')
    end = model_sdf_content.rfind('</model>')
    if start != -1 and end != -1:
        model_sdf_content = model_sdf_content[start:end+8]  # 8 is len('</model>')
    # ground plane内容
    ground_plane = '''
    <model name='ground_plane'>
      <static>true</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0.0 0.0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
            <surface>
                <friction>
                    <ode />
                </friction>
                <bounce />
                <contact />
            </surface>
        </collision>
        <visual name='visual'>
            <geometry>
                <plane>
                    <normal>0 0 1</normal>
                    <size>100 100</size>
                </plane>
            </geometry>
            <material>
                <ambient>0.8 0.8 0.8 1</ambient>
                <diffuse>0.8 0.8 0.8 1</diffuse>
                <specular>0.8 0.8 0.8 1</specular>
            </material>
            <plugin name='__default__' filename='__default__' />
        </visual>
      </link>
      <plugin name='__default__' filename='__default__' />
      <pose>0 0 0 0 0 0</pose>
    </model>'''
    # 生成world内容
    world_content = f'''<?xml version="1.0"?>
    <sdf version="1.8">
        <world name="{world_name}">
            <physics name='1ms' type='ignored'>
                <max_step_size>0.003</max_step_size>
                <real_time_factor>1</real_time_factor>
                <real_time_update_rate>1000</real_time_update_rate>
            </physics>
            <plugin name='ignition::gazebo::systems::Physics' filename='ignition-gazebo-physics-system' />
            <plugin name='ignition::gazebo::systems::UserCommands' filename='ignition-gazebo-user-commands-system' />
            <plugin name='ignition::gazebo::systems::SceneBroadcaster' filename='ignition-gazebo-scene-broadcaster-system' />
            <plugin name='ignition::gazebo::systems::Contact' filename='ignition-gazebo-contact-system' />
            <light name='sun' type='directional'>
                <cast_shadows>1</cast_shadows>
                <pose>0 0 10 0 -0 0</pose>
                <diffuse>0.8 0.8 0.8 1</diffuse>
                <specular>0.2 0.2 0.2 1</specular>
                <attenuation>
                    <range>1000</range>
                    <constant>0.90000000000000002</constant>
                    <linear>0.01</linear>
                    <quadratic>0.001</quadratic>
                </attenuation>
                <direction>-0.5 0.1 -0.9</direction>
                <spot>
                    <inner_angle>0</inner_angle>
                    <outer_angle>0</outer_angle>
                    <falloff>0</falloff>
                </spot>
            </light>
            <gravity>0 0 -9.8</gravity>
            <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
            <atmosphere type='adiabatic' />
            <scene>
                <ambient>0.4 0.4 0.4 1</ambient>
                <background>0.7 0.7 0.7 1</background>
                <shadows>1</shadows>
            </scene>
            {ground_plane}
            {model_sdf_content}
        </world>
    </sdf>'''.format(world_name=world_name)
    # 确保world输出目录存在
    world_file = os.path.join(world_folder, world_name + '.sdf')
    os.makedirs(world_folder, exist_ok=True)
    with open(world_file, "w") as f:
        f.write(world_content)
    print(f"world file saved to {world_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"用法: python {os.path.basename(__file__)} src_map_folder dst_model_folder dst_world_folder")
        sys.exit(1)
    map_folder = sys.argv[1]
    models_folder = sys.argv[2]
    world_folder = sys.argv[3]

    # 检查输入文件夹是否存在
    if not os.path.isdir(map_folder):
        print(f"地图文件夹不存在: {map_folder}")
        sys.exit(1)
    # 查找同名yaml和pgm文件
    files = os.listdir(map_folder)
    yaml_files = [f for f in files if f.endswith('.yaml')]
    pgm_files = [f for f in files if f.endswith('.pgm')]
    base_names = set(os.path.splitext(f)[0] for f in yaml_files) & set(os.path.splitext(f)[0] for f in pgm_files)
    if len(base_names) == 0:
        print("未找到同名的yaml和pgm文件，请确保输入文件夹内有同名的地图yaml和pgm文件。")
        sys.exit(1)

    print("maps found: ", base_names)
    for map in base_names:
        print(f"map founded: {map}")
        model_folder = os.path.join(models_folder, map)
        generate_model(map_folder, map, model_folder)
        generate_world(model_folder, world_folder, map)