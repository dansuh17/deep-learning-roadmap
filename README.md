# Deep Learning Roadmap
My own deep learning mastery roadmap, inspired by [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap).

**There are some customized differences:**
- not only academic papers but also blog posts, online courses, and other references are included
- customized for my own targeted roadmap - may not include RL, NLP, etc.
- updated for 2019 SOTA

### Introductory Courses
- [Deeplearning.ai courses by Andrew Ng](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w)
- [Fast.ai](https://www.fast.ai/)
- [Dive Into Deep Learning](http://d2l.ai/)
- [Stanford's CS231n Class Notes](http://cs231n.github.io/)
  
### Basic CNN Architectures
- [x] **AlexNet** (2012) [paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
    - Alex Krizhevsky et al. "ImageNet Classification with Deep Convolutional Neural Networks"
- [x] **ZFNet** (2013) [paper](https://arxiv.org/abs/1311.2901)
    - Zeiler et al. "Visualizing and Understanding Convolutional Networks"
- [x] **VGG**: (2014)
    - Simonyan et al. "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014) [Google DeepMind & Oxford] [[paper](https://arxiv.org/abs/1409.1556)]
    - _VGG-16_: Zhang et al. "Accelerating Very Deep Convolutional Networks for Classification and Detection" [[paper](https://arxiv.org/abs/1505.06798?context=cs)]
- [x] **ResNet**: https://arxiv.org/abs/1512.03385
- [x] [read] **GoogLeNet**: https://arxiv.org/abs/1409.4842
- [x] **Inception**: https://arxiv.org/abs/1512.00567
- [x] [read] **Xception**: https://arxiv.org/abs/1610.02357
- [x] **MobileNet**: https://arxiv.org/abs/1704.04861
- [ ] **DenseNet** (2018) [paper](https://arxiv.org/abs/1608.06993)


### Generative adversarial networks
- [x] GAN (2014.6) : https://arxiv.org/abs/1406.2661
- [x] DCGAN: (2015.11): https://arxiv.org/abs/1511.06434
- [x] Info GAN (2016.6) : https://arxiv.org/pdf/1606.03657.pdf
- [x] f-GAN (2016.6) : https://arxiv.org/abs/1606.00709
- [x] Unrolled GAN : (2016.7) : https://arxiv.org/abs/1611.02163
- [x] ACGAN (2016.10) : https://arxiv.org/abs/1610.09585
- [x] LSGAN (2016.11) : https://arxiv.org/abs/1611.04076
- [x] Pix2Pix (2016.11) [read] : https://arxiv.org/abs/1611.07004
- [x] EBGAN (2016.11) : https://arxiv.org/abs/1609.03126
- [x] WGAN (2017.4) : https://arxiv.org/abs/1701.07875
    - [x] WGAN_GP (2017.5) : https://arxiv.org/abs/1704.00028
- [x] BEGAN (2017.5) : https://arxiv.org/abs/1703.10717
- [ ] CycleGAN (2017.5) : https://arxiv.org/abs/1703.10593
- [ ] DiscoGAN (2017.5) : https://arxiv.org/abs/1703.05192
- [ ] DRAGAN (2017.5) : https://arxiv.org/abs/1705.07215
- [ ] ProGAN (2017.10) : https://arxiv.org/abs/1710.10196
- [ ] SAGAN (2018.5)  : https://arxiv.org/abs/1805.08318
- [ ] BigGAN (2018) : https://arxiv.org/abs/1809.11096


### advanced GANs
- [ ] Augmented CycleGAN
- [ ] Joint GAN
- [ ] Semi-Amortized Variational Autoencoders
- [ ] Self-Attention GAN : https://github.com/brain-research/self-attention-gan
- [ ] GAN dissection https://arxiv.org/abs/1811.10597v1
- [ ] Banach Wasserstein GAN https://arxiv.org/abs/1806.06621
- [ ] Are GANs created equal? https://arxiv.org/abs/1711.10337
- [ ] Semantic Image Synthesis with Spatially-Adaptive Normalization
 https://arxiv.org/abs/1903.07291
- [ ] Self-supervised GAN https://arxiv.org/abs/1811.11212
- [ ] [Which Training Methods for GAN do actually converge? (2018.7)](https://arxiv.org/abs/1801.04406)
- [ ] [Improving Generalization and Stability for GANs (2019)](https://openreview.net/pdf?id=ByxPYjC5KQ&fbclid=IwAR2_8Qft8cIX3y-Cki-4JzWMeoxm91yUq1ELA3N7eJBMTedPuUz8H6vvqMo)

### Autoencoders
- [ ] Original autoencoder (1986) [paper](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf)
    - Rumelhart, Hinton, and Williams, "Learning Internal Representations by Error Propagation"
- [x] **AutoEncoder** [science](https://www.cs.toronto.edu/~hinton/science.pdf)
    - Hinton et al., "Reducing the Dimensionality of Data with Neural Networks"
- [ ] Denoising Autoencoders (2008) [paper](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
    - Vincent et al. "Extracting and Composing Robust Features with Denoising Autoencoders"
- [ ] sliced wasserstein autoencoders : https://arxiv.org/abs/1804.01947

### Autoregressive models
- [ ] PixelCNN (2016) [paper](https://arxiv.org/abs/1606.05328)
    - Oord, Aaron van den, et al. "Conditional image generation with PixelCNN decoders."
- [ ] WaveNet
- [ ] tacotron

### Stochastic gradient descent techniques
- [On large batch training of NN](https://arxiv.org/abs/1609.04836)

### Layer Normalizations
- Batch Norm
- Group Norm
- Instance Norm
- **Switchable Normalization** (2019) [[paper](https://arxiv.org/abs/1806.10779)]
    - Luo et al. "Differentiable Learning-to-Normalize via Switchable Normalization"
- Weight Standardization : https://arxiv.org/abs/1903.10520

### Dropouts
- **Dropout** (2014) [[paper](http://jmlr.org/papers/v15/srivastava14a.html)]
    - Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Inverted Dropouts** [[notes on CS231n](http://cs231n.github.io/neural-networks-2/#reg)]
    - Multiplying the inverted _keep_prob_ value on training so that values during inference (or testing) is consistent.

### Zero-shot learning
### Transfer learning
### Representation learning
- Neural Discrete Representation Learning : https://arxiv.org/pdf/1711.00937.pdf

### Variational Autoencoders (VAE)
### Neural architecture search and AutoML
- Efficient Neural Architecture Search by Parameter Sharing : https://arxiv.org/pdf/1802.03268.pdf
- AdaNet
- **Randomly Wired Neural Nets** (2019) [[paper](https://arxiv.org/abs/1904.01569)]
    - Xie et al. "Exploring Randomly Wired Neural Networks for Image Recognition" (Facebook AI Research)

### Object detection
- RCNN: https://arxiv.org/abs/1311.2524
- Fast-RCNN: https://arxiv.org/abs/1504.08083
- Faster-RCNN: https://arxiv.org/abs/1506.01497
- SSD: https://arxiv.org/abs/1512.02325
- [ ] YOLO: https://arxiv.org/abs/1506.02640
- YOLO9000: https://arxiv.org/abs/1612.08242

### Semantic Segmentation
- [ ] FCN: https://arxiv.org/abs/1411.4038
- [ ] SegNet: https://arxiv.org/abs/1511.00561
- [ ] UNet: https://arxiv.org/abs/1505.04597
- [ ] PSPNet: https://arxiv.org/abs/1612.01105
- [ ] DeepLab: https://arxiv.org/abs/1606.00915
- [ ] ICNet: https://arxiv.org/abs/1704.08545
- [ ] ENet: https://arxiv.org/abs/1606.02147
- [Nice survey](https://github.com/hoya012/deep_learning_object_detection)

### Sequential Model
- [ ] Seq2Seq [36] Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems. 2014. [pdf] (Outstanding Work) ⭐️⭐️⭐️⭐️⭐️

### Neural Turing Machine
- [ ] **Neural Turing Machines** (2014) [[paper](https://arxiv.org/abs/1410.5401)]
    - Graves et al., "Neural turing machines."
- [ ] **Pointer Networks** (2015) [[paper]](https://arxiv.org/abs/1506.03134)]
    - Vinyals et al., "Pointer networks."

### Attention
- [ ] **NMT (Neural Machine Translation)** (2014) [[paper](https://arxiv.org/abs/1409.0473)]
    - Bahdanau et al, "Neural Machine Translation by Jointly Learning to Align and Translate"
- [ ] **Attention is all you need** (2017) [[paper](https://arxiv.org/abs/1706.03762)]
     -  Vaswani et al. "Attention is all you need"    
- [ ] [read] Lilian Weng - **"Attention? Attention!"** (2018) [[blog_post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)]
    - A nice explanation of attention mechanism and its concepts.

### Advanced RNNs
- [ ] Unitary evolution RNNs : https://arxiv.org/abs/1511.06464
- [ ] Recurrent Batch Norm : https://arxiv.org/abs/1603.09025
- [ ] Zoneout : https://arxiv.org/abs/1606.01305
- [ ] IndRNN : https://arxiv.org/abs/1803.04831
- [ ] DilatedRNNs : https://arxiv.org/abs/1710.02224

### Neural Processes
- [ ] **Neural Processes** (2018) [[paper](https://arxiv.org/abs/1807.01622)]
    - Garnelo et al. "Neural Processes"
- [ ] **Attentive Neural Processes** (2019) [[paper](https://arxiv.org/abs/1901.05761)]
    - Kim et al. "Attentive Neural Processes"
- [ ] **A Visual Exploration of Gaussian Processes** (2019) [[Distill.pub](https://distill.pub/2019/visual-exploration-gaussian-processes/)]
    - Not a neural process, but gives very nice intuition about Gaussian Processes. Good Read.

### Self-supervised learning

- [x] Denoising AE https://www.iro.umontreal.ca/~vincentp/Publications/denoising_autoencoders_tr1316.pdf
- [ ] Exemplar Nets https://arxiv.org/abs/1406.6909
- [ ] Co-occ https://arxiv.org/abs/1511.06811
- [ ] Egomotion https://arxiv.org/abs/1505.01596
- [ ] Jigsaw https://arxiv.org/abs/1603.09246
- [ ] Context Encoders https://arxiv.org/abs/1604.07379
- [ ] Split-brain autoencoders https://arxiv.org/abs/1611.09842
- [ ] multi-task self-supervised learning https://arxiv.org/abs/1708.07860
- [ ] Audio-visual scene analysis https://arxiv.org/abs/1804.03641
- [ ] a survey https://slideplayer.com/slide/13195863/
- [ ] Supervising unsupervised learning https://arxiv.org/abs/1709.05262
- [ ] Unsupervised Representation Learning by Predicting Image Rotations
 https://arxiv.org/abs/1803.07728

### Interpretability
- [ ] Visualizing loss landscape of neural nets (2018) https://arxiv.org/abs/1712.09913v3

## DL roadmap reference
- https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap
- https://github.com/terryum/awesome-deep-learning-papers
- which DL algorithms should I implement to learn?
https://www.reddit.com/r/MachineLearning/comments/8vmuet/d_what_deep_learning_papers_should_i_implement_to/

### Theory
- [Theoretical principles for Deep Learning](http://mitliagkas.github.io/ift6085-dl-theory-class-2019/?fbclid=IwAR02mfw0hMM5UFaBuxBgDi5wfT6TaIt35wktZGpCmmhu0_3GMA6HqFN1GFs)
- [Bubeck - Convex Optimization - Algorithms and Complexity](https://arxiv.org/pdf/1405.4980.pdf)
- [Stanford STATS 385 - Theories of Deep Learning](https://stats385.github.io/)
    - [Lecture Videos](https://www.researchgate.net/project/Theories-of-Deep-Learning?fbclid=IwAR0dwnuswA1jMwIOuydb_a83AM22FfuD6PpAWPiIW-76OCemcRBrVVLKLoM)

## References
- [ ] CSC 231 notes : http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/
