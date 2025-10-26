# -*- coding: utf-8 -*
#import os
#import radiomics.featureextractor as FEE

#import csv

import numpy as np
import pandas as pd
import radiomics
from radiomics import featureextractor

sample_list = np.loadtxt('/dataset_all.txt', delimiter=None, dtype=str)
nrrd_Path = '/data_nsclc_standardized/CT/'
seg_Path = '/data_nsclc_standardized/CT-seg/'


# 定义特征提取设置
settings = {}
settings['binWidth'] = 25
settings['sigma'] = [3, 5]
# 图像/mask重采样：
settings['resampledPixelSpacing'] = [1,1,1]   #设置重采样时的体素大小  # 3,3,3
settings['voxelArrayShift'] = 1000     # 300

#图像归一化：
settings['normalize'] = True           #图像归一化
settings['normalizeScale'] = 100       #图像归一化后的比例

# 实例化特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

# 指定用于提取特征的图像类型
# 指定使用 LoG 和 Wavelet 滤波器
extractor.enableImageTypeByName('LoG')
extractor.enableImageTypeByName('Wavelet')
# 所有类型
extractor.enableAllFeatures()
extractor.enableFeaturesByName(firstorder=['Energy', 'TotalEnergy', 'Entropy','Minimum', '10Percentile', '90Percentile',
                                                 'Maximum', 'Mean', 'Median', 'InterquartileRange', 'Range',
                                                 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation','RootMeanSquared',
                                                 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
extractor.enableFeaturesByName(shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2', 
                                            'Sphericity', 'SphericalDisproportion',  'Maximum3DDiameter', 'Maximum2DDiameterSlice', 
                                            'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 
                                            'MajorAxisLength', 'MinorAxisLength', 'LeastAxisLength', 'Elongation', 'Flatness'])


features_dict = dict()
df = pd.DataFrame()

for i, (name, label) in enumerate(sample_list):
    imagePath = nrrd_Path + name + '.nrrd'
    maskPath = seg_Path + name + '_seg.nrrd'
    print(str(i) + '/325 | ' + name)
    features = extractor.execute(imagePath, maskPath)  #抽取特征
    
    for key, value in features.items():  #输出特征
            features_dict[key] = value
    
    df = df.append(pd.DataFrame.from_dict(features_dict.values()).T,ignore_index=True)
    
df.columns = features_dict.keys()
df.to_csv('/Radiomics-Features.csv', index=0)
print('Done')












































'''
para_name = '/media/user/Disk04/gaoheng/code/exampleCT.yaml'

# clinical_data = np.array(pd.read_csv('/media/user/Disk02/lizhe/lung/data/NSCLC/nsclc_select.csv', encoding="GB2312"))
# pat_name = clinical_data[:, 0]

img_name = os.listdir(ori_path)
label_names = os.listdir(lab_path)

img_name = sorted(img_name, key=lambda x: (x.split('.nrrd')[-2]))
label_names = sorted(label_names, key=lambda x: (x.split('_seg.nrrd')[-2]))

# print(img_name)
# print(label_names)

for i in range(0, len(img_name)):
    pat_name = img_name[i].split('.')[-2]
    # 文件全部路径
    image = ori_path + img_name[i]
    label = lab_path + label_names[i]

    para_path = para_name
    # print("originl path: " + ori_path)
    # print("label path: " + lab_path)
    # print("parameter path: " + para_path)

    # 使用配置文件初始化特征抽取器
    extractor = FEE.RadiomicsFeatureExtractor(para_path)
    # 运行
    result = extractor.execute(image, label)  # 抽取特征
    print("Result type:", type(result))  # result is returned in a Python ordered dictionary
    print("")
    print("Calculated get_features")
    
    with open('get_features/NSCLC_509/' + pat_name + ".csv", "w", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["Type", pat_name])
        for key, values in result.items():  # 输出特征
            writer.writerow([key, values])
'''