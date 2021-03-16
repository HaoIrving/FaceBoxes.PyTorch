# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_plain/
# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_nl/ -nl
# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_se/ -se
# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_cbam/ -cbam
# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_gcb/  -gcb
# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_cda/  -cda

# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_plain/
# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_nl/ -nl
# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_se/ -se
# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_cbam/ -cbam
# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_gcb/  -gcb
# CUDA_VISIBLE_DEVICES=2,3  python train.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_cda/  -cda


# CUDA_VISIBLE_DEVICES=2  python test.py --prefix weights/lr_1e3_plain 
# CUDA_VISIBLE_DEVICES=0  python test.py --prefix weights/lr_1e3_nl    -nl
# CUDA_VISIBLE_DEVICES=1  python test.py --prefix weights/lr_1e3_se    -se
# CUDA_VISIBLE_DEVICES=3  python test.py --prefix weights/lr_1e3_cbam  -cbam
# CUDA_VISIBLE_DEVICES=2  python test.py --prefix weights/lr_1e3_gcb   -gcb
# CUDA_VISIBLE_DEVICES=3  python test.py --prefix weights/lr_1e3_cda   -cda

######### light ############

# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception/  -x
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception/  -x
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_xception  -x

# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception_DscCRelu/  -dcr
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception_DscCRelu/  -dcr
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_xception_DscCRelu  -dcr


# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception_mbv2/  -mb
python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception_mbv2/  -mb
CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_xception_mbv2  -mb

# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception_shfv2/  -shf
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_xception_shfv2  -shf
