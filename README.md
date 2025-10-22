# use this command to check for the quick test if the model is learning or not then proceed for training
python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_quick_test -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --epochs 30 --subset 0.1 --resume checkpoint/SST_Model_quick_test/best_epoch

## PLEASE MAKE SURE TO MAKE SOME CHANGES IN DATASET LOADER AND DATASET READER IF YOU WANT TO , DUW TO CURROPT DATA OF SOME SUBJECTS I SKIPPED THOSE SUBJECTS BY HARCODING THE SKIPPING SEQUENCE IN THE CODE { IT MAY OR MAY NOT LOAD WHOLE DATA } 

# use this command to train the model for the 1st epoch
python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_final -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --epoch-size 1000 -lr 5e-4


# use this command to resume your training after saving a checkpoint
python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_final -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --resume checkpoint/SST_Model_final/best_epoch.bin  --epoch-size 1000 -lr 5e-4

# Resume training for 81 frame sequence after saving a checkpoint
 python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_81_final -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4 -b 10 --resume checkpoint/SST_Model_81_final/best_epoch.bin  --epoch-size 1000 -lr 5e-4



# use this command to evaluate the model for 27 fram sequence
python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb -frame 27 --evaluate checkpoint/SST_Model_final/best_epoch.bin


# use this command to evaluate the model 81 frames
python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --evaluate checkpoint/SST_Model_81_final/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81 --subjects-test "S9" --epoch-size 1000 -b 5


# use this to run interactive visualisation of the detection before fine tuning
python final_demo.py --video path/to/video.mp4 --checkpoint-3d checkpoint/SST_Model_final/best_epoch.bin -frame 27 -frame-kept 27 -coeff-kept 27 

# use this to run interactive visualisation of the detection for 81 frames before fine tuning
python final_vis.py --video path/to/video.mp4 --checkpoint-3d checkpoint/SST_Model_81_final/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81


# use this to fine tune for the 1st Epoch

python finetune.py -g 0 --dataset merged --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_Finetuned --subjects-train "S1,S5,S6,S7,S8,S3_MPI,S4_MPI,S5_MPI,S6_MPI,S7_MPI" --subjects-test "S8_MPI" --resume checkpoint/SST_Model_81_final/best_epoch.bin -lr 2e-5 --epoch-size 1000 -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4 -b 10

# use this comand for resuming the fine tuning of the model 
python finetune.py -g 0 --dataset merged --keypoints cpn_ft_h36m_dbb --checkpoint  "checkpoint/SST_Model_Finetuned" --subjects-train "S1,S5,S6,S7,S8,S3_MPI,S4_MPI,S5_MPI,S6_MPI,S7_MPI" --subjects-test "S8_MPI" --resume checkpoint/SST_Model_Finetuned/best_epoch.bin -lr 2e-5 --epoch-size 1000 -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4 -b 10

## use this to run interactive visualisation of the detection for 81 frames after fine tuning
python final_vis.py --video path/to/video.mp4 --checkpoint-3d checkpoint/SST_Model_Finetuned/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81




# it will make a video

python vis.py --video path/to/your/video.mp4 --checkpoint-3d checkpoint/SST_Model_81_final/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4


python vis.py --video input\sample.mp4 --checkpoint-3d checkpoint/SST_Model_Finetuned/best_epoch.bin -frame 81 -frame-kept 81 -coeff-kept 81 --depth 4


<video controls src="input/dance-m.mp4" title="Title"></video>