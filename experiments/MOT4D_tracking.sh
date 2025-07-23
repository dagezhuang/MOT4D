cd src

python -u main.py tracking --modal VM --save_all --exp_id MOT4D --dataset mot_rgbt --dataset_version SpaceAnimals --load_model /home/ubuntu/exp/pretrain/model.pth --batch_size 8 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0,1


# train, the model is finetuned from a CenterNet detection model from the CenterNet model zoo.
# python main.py tracking --exp_id coco_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth  --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1