cd src
CUDA_VISIBLE_DEVICES=3 python test_rgbt.py tracking --modal VM --test_mot4d True --exp_id MOT4D_fusion_mid --dataset SpaceAnimals --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model /home/ubuntu/exp/tracking/MOT4D_fusion_mid/model.pth
cd ..