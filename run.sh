# python train.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_plain/
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


# CUDA_VISIBLE_DEVICES=2  python test.py --prefix weights/lr_1e3_3ach_plain 
# CUDA_VISIBLE_DEVICES=0  python test.py --prefix weights/lr_1e3_nl    -nl
# CUDA_VISIBLE_DEVICES=1  python test.py --prefix weights/lr_1e3_se    -se
# CUDA_VISIBLE_DEVICES=3  python test.py --prefix weights/lr_1e3_cbam  -cbam
# CUDA_VISIBLE_DEVICES=2  python test.py --prefix weights/lr_1e3_gcb   -gcb
# CUDA_VISIBLE_DEVICES=3  python test.py --prefix weights/lr_1e3_cda   -cda

######### light + att ##################
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2_nl/  -nl
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_nl  -nl

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2_se/  -se
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_se  -se 
# # CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_se  -se --cpu

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2_cbam/  -cbam
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_cbam  -cbam

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2_gcb/  -gcb
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_gcb  -gcb

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2_cda/  -cda
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2_cda  -cda


######### light ############
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3/  -x
# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3/  -x
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3  -x

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception3_xcp_3ach/  -xcp
# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_xception3_xcp_3ach/  -xcp
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_xception3_xcp_3ach  -xcp


# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception_mbv2/  -mb
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception_mbv2/  -mb
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception_mbv2  -mb

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv2/  -mb
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2  -mb
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv2  -mb --cpu --eepoch 250

# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_mbv1/  -mbv1
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv1  -mbv1
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_mbv1  -mbv1 --cpu --eepoch 285

# CUDA_VISIBLE_DEVICES=2,3  python train_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_shfv2/  -shf
# python train_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_shfv2/  -shf
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_shfv2  -shf
# CUDA_VISIBLE_DEVICES=2  python test_light.py --prefix weights/lr_1e3_3ach_xception3_shfv2  -shf  --cpu --eepoch final

######### dsc + light ###########
# python train_dsc_light.py --lr 1e-3 --ngpu 4 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_dsc_xception3_3ach/  -x
# CUDA_VISIBLE_DEVICES=2,3  python train_mbv2_light.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_mbv2_xception3_3ach/  -x
# CUDA_VISIBLE_DEVICES=2  python test_mbv2_light.py --prefix weights/lr_1e3_mbv2_xception3_3ach  -x

######### light + xcp + att#######
# CUDA_VISIBLE_DEVICES=2,3 python train_light_xcp.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_xcp_nl/  -nl
# CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_nl  -nl
CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_nl  -nl --cpu --eepoch 230

# CUDA_VISIBLE_DEVICES=2,3 python train_light_xcp.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_xcp_se/  -se
CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_se  -se  --eepoch 245 --show_image
# CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_se  -se --cpu

# CUDA_VISIBLE_DEVICES=2,3 python train_light_xcp.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_xcp_cbam/  -cbam
# CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_cbam  -cbam
CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_cbam  -cbam --cpu --eepoch 290

# CUDA_VISIBLE_DEVICES=2,3 python train_light_xcp.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_xcp_gcb/  -gcb
# CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_gcb  -gcb
CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_gcb  -gcb --cpu --eepoch 255

# CUDA_VISIBLE_DEVICES=2,3 python train_light_xcp.py --lr 1e-3 --ngpu 2 --num_workers 12 -b 32 --save_folder ./weights/lr_1e3_3ach_xception3_xcp_cda/  -cda
# CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_cda  -cda
CUDA_VISIBLE_DEVICES=2  python test_light_xcp.py --prefix weights/lr_1e3_3ach_xception3_xcp_cda  -cda --cpu --eepoch 220
