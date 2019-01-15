import os
import json
# data_path = './classes/outputs'
user_home = os.getenv('HOME')
data_path = user_home + "/Desktop/img"
test_data_path = user_home + '/Desktop/testimg/test'

images = []
labels = []
test_images = []
test_labels = []
"""
for jsonfile in os.listdir(data_path):
    if jsonfile.endswith("json"):
        with open(os.path.join(data_path + "/", jsonfile), 'r') as jf:
            map_json = json.load(jf)
            # print(map_json)
            images.append("." + map_json['path'].split("Desktop")[1].replace('\\', '/'))
            labels.append(map_json['outputs']['transcript'])
            # print(jsonfile)
"""

for img_file in os.listdir(test_data_path):
    if img_file.endswith(".jpeg"):
        test_images.append(os.path.join(test_data_path + "/", img_file))
        test_labels.append(os.path.splitext(img_file)[0])

        # print(test_images)
        # print(test_labels)


for img_file in os.listdir(data_path):
    if img_file.endswith(".jpeg"):
        images.append(os.path.join(data_path + "/", img_file))
        labels.append(os.path.splitext(img_file)[0])

        # print(images)
        # print(labels)
