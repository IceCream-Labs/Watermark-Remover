K_CENTER=2
K_REFINE=3
K_SKIP=3
MASK_MODE=res
INPUT_SIZE=256
NAME=slbr_v1

#NOT TO CHANGE ANYTHING ABOVE, UPDATE ONLY THE BELOW DIRECTORIES AND LINKS ACCORDINGLY
#BASNET IS THE SEGMENTATION MODULE AND SLBR IS FOR WATERMARK REMOVAL, INFERENCE DIR IS PATH OF INFERENCE.py AND TEST_DIR IS PATH OF #INFERENCE IMAGES

TEST_DIR=/home/icecreamlabs/watermarkremoval/Final_test_images/
INFERENCE_DIR=/home/icecreamlabs/watermarkremoval/INFERENCE.py
BASNET_LINK='https://github.com/KKamaleshKumar/WMR-Model-Weigths/releases/download/BASNet/basnet.pth'
SLBR_LINK='https://github.com/KKamaleshKumar/WMR-Model-Weigths/releases/download/BASNet/model_best.pth.tar'
ONLY_BG_REMOVAL=$1      #True or False (case sensitive)



python3  ${INFERENCE_DIR} \
  --name ${NAME} \
  --nets slbr \
  --models slbr \
  --input-size ${INPUT_SIZE} \
  --crop_size ${INPUT_SIZE} \
  --test-batch 1 \
  --evaluate\
  --preprocess resize \
  --no_flip \
  --mask_mode ${MASK_MODE} \
  --k_center ${K_CENTER} \
  --use_refine \
  --k_refine ${K_REFINE} \
  --k_skip_stage ${K_SKIP} \
  --test_dir ${TEST_DIR} \
  --resume ${SLBR_LINK} \
  --basnet_weights ${BASNET_LINK} \
  --bg_remove_only ${ONLY_BG_REMOVAL} \
 
