python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb --checkpoint checkpoint/SST_Model_final -frame 27 -frame-kept 27 -coeff-kept 27 --depth 4 -b 16 --resume checkpoint/SST_Model_final/best_epoch.bin 
  # use this command to train the model 



python run_SST.py -g 0 --dataset h36m --keypoints cpn_ft_h36m_dbb -frame 27 --evaluate checkpoint/SST_Model_final/best_epoch.bin
# use this command to evaluate the model
 




