## Instructions for Inference

BasNet (segmentation Module) Weights:  https://github.com/KKamaleshKumar/WMR-Model-Weigths/releases/download/BASNet/basnet.pth

SLBR (Water Mark Removal) Weights:  https://github.com/KKamaleshKumar/WMR-Model-Weigths/releases/download/BASNet/model_best.pth.tar

FeMaSR (for 4X super resoution) : https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth

FeMaSR (for 2X super resolution) :  https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth



All weights will load automatically when model is called for inference.

For inference run INFERENCE.sh and specify path to INFERENCE.py file

Specify images directory for testing using TEST_DIR. Make sure images are placed in a separate folder inside the TEST_DIR. Because two new folders TEST_DIR/pass_to_segment  and TEST_DIR/final_output will be created at the end of inference. The final_output folder holds the pipeline outputs.

The  ONLY_BG_REMOVAL argument takes two strings either 'True' or 'False' and is case-sensitive. True if only background removal is required.
