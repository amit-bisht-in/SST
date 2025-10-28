python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_final -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --resume checkpoint/SST_Model_final/best_epoch.bin 
  # use this command to train the model 



python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb -frame 27 --evaluate checkpoint/SST_Model_final/best_epoch.bin
# use this command to evaluate the model
 




python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb -frame 27 --evaluate checkpoint/SST_Model_final/best_epoch.bin




python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_quick_test -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --epochs 30 --subset 0.1 --resume checkpoint/SST_Model_quick_test/best_epoch


python final_vis.py --video input\test.mp4 --checkpoint-3d checkpoint/SST_Model_final/best_epoch.bin   

<<<<<<< Updated upstream
# use this to run interactive visualisation of the detection
=======
python finetune.py -g 0 --dataset merged  --checkpoint checkpoint/SST_Model_Finetuned --resume checkpoint/SST_Model_27/best_epoch.bin -lr 2e-5 --epoch-size 1000 -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 10

# use this comand for resuming the fine tuning of the model 
python finetune.py -g 0 --dataset merged  --checkpoint checkpoint/SST_Model_Finetuned --resume checkpoint/SST_Model_Finetuned/best_epoch.bin -lr 2e-5 --epoch-size 1000 -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 10

## use this to run interactive visualisation of the detection for 81 frames after fine tuning
python final_vis.py --video path/to/video.mp4 --checkpoint-3d checkpoint/SST_Model_Finetuned/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81




# it will make a video

python vis.py --video path/to/your/video.mp4 --checkpoint-3d checkpoint/SST_Model_81_final/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4


python vis.py --video input\sample.mp4 --checkpoint-3d checkpoint/SST_Model_Finetuned/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4


<video controls src="input/dance-m.mp4" title="Title"></video>
>>>>>>> Stashed changes
