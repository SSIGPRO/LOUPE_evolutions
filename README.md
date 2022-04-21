# LOUPE_evolutions

"A Deep Learning Method for Optimal Undersampling Patterns and Image Recovery for MRI Exploiting Losses and Projections", Filippo Martinini, Mauro Mangia, Alex Marchioni, Riccardo Rovatti, and Gianluca Setti.

--- Intro

Compressed Sensing was recently proposed to reduce the long acquisition time of Magnetic Resonance Imaging by undersampling the signal frequency content and then algorithmically reconstructing the original image.
We propose a way to significantly improve the classic fast MRI acquisition by exploiting a deep neural network to tackle both problems of frequency sub-sampling and image reconstruction simultaneously, thanks to the introduction of a new loss function to drive the training and the addition of a post-processing non-neural stage. 
All improvements hinge on the possibility of identifying constraints to which the final image must obey and suitably enforce them.

--- Description

Our work is based and evolves the models presented in:  

1. "Deep-learning-based Optimization of the Under-sampling Pattern in MRI" C. Bahadir‡, A.Q. Wang‡, A.V. Dalca, M.R. Sabuncu. IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020. doi: 10.1109/TCI.2020.3006727.
2. "Learning-based Optimization of the Under-sampling Pattern in MRI". Cagla D. Bahadir, Adrian V. Dalca, and Mert R. Sabuncu. IPMI: Information Processing in Medical Imaging. 2019. DOI: https://doi.org/10.1007/978-3-030-20351-1_61

In particular, we refer to the original LOUPE model as "dec0|L0". It consists of an encoder and a decoder. The encoder undersamples a fully sampled MRI scan in the frequency domain and brings it back to the spatial domain. For this purpose the encoder trains a binary mask that decides which k-space samples of the MRI to take or to discard (given a fixed number of elements to keep). The decoder ,instead, reconstructs the discarded frequency by using a U-NET, that takes the undersampled image in the spatial domain and returns a restored version of it. The loss "L0", that computes the mean absolute error (mae) between the ground truth and the reconstructed image, is minimized during training to optimize the model.


We propose:
1. dec1|L1 / dec1*|L1, where dec1 introduces a slight modification of the U-NET structure to enable its output to have two features, and L1 introduces a regularization term to L0 that compares the encoded reconstructed k-space with the acquired k-space. 
2. dec1|L2 / dec1*|L2, where L2 adds a regularization term to L0 that weights the difference between the non-acquired elements of the k-space of the reconstructed image and the same elements of the label image (non-sensed k-space). In practice, we noticed that best results are returned by setting the weight of the regularization to 1 (and the other to 0).
3. dec2|L0 / dec2*|L0, where dec2 sees a new projection block attached to dec1, that forces the reconstructed image to be compliant with the measurament constraint.

Here * indicates that a post-training processing stage has been used to improve the final reconstruction. This processing stage, by iteratively applying the Dykstra's projection algorithm, finds the real-valued image belonging to the set of images with k-space matching the measurements that is closer to the reconstructed image.

---

Inside modules:
1. "models.py" contains the implementation of all the dec(i)|Lj (for i,j in {0,1,2}).
2. "layers.py" implements all the custom structure that build the models.
3. "utility.py" stores all the functions used to manipulate the models.


