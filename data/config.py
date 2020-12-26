# config.py

# cfg = {
#     'name': 'FaceBoxes',
#     #'min_dim': 1024,
#     #'feature_maps': [[32, 32], [16, 16], [8, 8]],
#     'aspect_ratios': [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]],
#     # 'min_sizes': [[8, 16], [32, 64, 128], [256], [512]],
#     'min_sizes': [[8, 16], [23, 28], [36, 47, 57], [68, 79, 89], [113, 129], [207, 512]],
#     'steps': [4, 8, 16, 32, 64, 128],

#     # 'aspect_ratios': [[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]],
#     # 'aspect_ratios': [[1], [1], [1]],
#     # 'min_sizes': [[32, 64, 128], [256], [512]],
#     # 'steps': [32, 64, 128],

#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 2.0,
#     'gpu_train': True
# }

cfg = {
    'name': 'FaceBoxes',
    #'min_dim': 1024,
    #'feature_maps': [[32, 32], [16, 16], [8, 8]],
    # 'aspect_ratios': [[1], [1], [1]],
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}
