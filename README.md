# Deep Learning Roadmap
My own deep learning mastery roadmap, inspired by [Deep Learning Papers Reading Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap).

**There are some customized differences:**
- not only academic papers but also blog posts, online courses, and other references are included
- customized for my own plans - may not include RL, NLP, etc.
- updated for 2019 SOTA

### Introductory Courses
- [Deeplearning.ai courses by Andrew Ng](https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w)
- [Fast.ai](https://www.fast.ai/)
- [Dive Into Deep Learning](http://d2l.ai/)
- [Stanford's CS231n Class Notes](http://cs231n.github.io/)
  
### Basic CNN Architectures
- [x] **AlexNet** (2012) [[paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)]
    - Alex Krizhevsky et al. "ImageNet Classification with Deep Convolutional Neural Networks"
- [x] **ZFNet** (2013) [[paper](https://arxiv.org/abs/1311.2901)]
    - Zeiler et al. "Visualizing and Understanding Convolutional Networks"
- [x] **VGG** (2014)
    - Simonyan et al. "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014) [Google DeepMind & Oxford's Visual Geometry Group (VGG)] [[paper](https://arxiv.org/abs/1409.1556)]
    - _VGG-16_: Zhang et al. "Accelerating Very Deep Convolutional Networks for Classification and Detection" [[paper](https://arxiv.org/abs/1505.06798?context=cs)]
- [x] **GoogLeNet**, a.k.a **Inception v.1** (2014) [[paper](https://arxiv.org/abs/1409.4842)]
    - Szegedy et al. "Going Deeper with Convolutions" [Google]
    - Original [LeNet page](http://yann.lecun.com/exdb/lenet/) from Yann LeCun's homepage.
    - [x] **Inception v.2 and v.3** (2015) Szegedy et al. "Rethinking the Inception Architecture for Computer Vision" [[paper](https://arxiv.org/abs/1512.00567)]
    - [x] **Inception v.4 and InceptionResNet** (2016) Szegedy et al. "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" [[paper](https://arxiv.org/abs/1602.07261)]
    - "A Simple Guide to the Versions of the Inception Network" [[blogpost](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)]
- [x] **ResNet** (2015) [[paper](https://arxiv.org/abs/1512.03385)]
    - He et al. "Deep Residual Learning for Image Recognition"
- [x] **Xception** (2016) [[paper](https://arxiv.org/abs/1610.02357)]
    - Chollet, Francois - "Xception: Deep Learning with Depthwise Separable Convolutions"
- [x] **MobileNet** (2016) [[paper](https://arxiv.org/abs/1704.04861)]
    - Howard et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    - A nice paper about reducing CNN parameter sizes while maintaining performance.
- [x] **DenseNet** (2016) [[paper](https://arxiv.org/abs/1608.06993)]
    - Huang et al. "Densely Connected Convolutional Networks"


### Generative adversarial networks
- [x] **GAN** (2014.6) [[paper](https://arxiv.org/abs/1406.2661)]
    - Goodfellow et al. "Generative Adversarial Networks"
- [x] **DCGAN** (2015.11) [[paper](https://arxiv.org/abs/1511.06434)]
    - Radford et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"
- [x] **Info GAN** (2016.6) [[paper](https://arxiv.org/abs/1606.03657)]
    - Chen et al. "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"
- [x] **Improved Techinques for Training GANs** (2016.6) [[paper](https://arxiv.org/abs/1606.03498)]
    - Salimans et al. "Improved Techinques for Training GANs"
    - This paper suggests multiple GAN training techinques such as _feautre matching_, _minibatch discrimination_, _one sided label smoothing_, _virtual batch normalization_.
    - It also suggests a renown generator performance metric, called the **inception score**.
- [x] **f-GAN** (2016.6) [[paper](https://arxiv.org/abs/1606.00709)]
    - Nowozin et al. "f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization"
- [x] **Unrolled GAN** (2016.7) [[paper](https://arxiv.org/abs/1611.02163)]
    - Metz et al. "Unrolled Generative Adversarial Networks"
- [x] **ACGAN** (2016.10) [[paper](https://arxiv.org/abs/1610.09585)]
    - Odena et al. "Conditional Image Synthesis With Auxiliary Classifier GANs"
- [x] **LSGAN** (2016.11) [[paper](https://arxiv.org/abs/1611.04076)]
    - Mao et al. "Least Squares Generative Adversarial Networks"
- [x] **Pix2Pix** (2016.11) [[paper](https://arxiv.org/abs/1611.07004)]
    - Isola et al. "Image-to-Image Translation with Conditional Adversarial Networks"
- [x] **EBGAN** (2016.11) [[paper](https://arxiv.org/abs/1609.03126)]
    - Zhao et al. "Energy-based Generative Adversarial Network"
- [x] **WGAN** (2017.4) [[paper](https://arxiv.org/abs/1701.07875)]
    - Arjovsky et al., "Wasserstein GAN"
- [x] **WGAN_GP** (2017.5) [[paper](https://arxiv.org/abs/1704.00028)]
    - Gulrajani et al., "Improved Training of Wasserstein GANs"
    - Improves the training stability by applying **"gradient penalty (GP)"** to the loss function
- [x] **BEGAN** (2017.5) [[paper](https://arxiv.org/abs/1703.10717)]
    - Berthelot et al. "BEGAN: Boundary Equilibrium Generative Adversarial Networks"
    - Introduces a _diversity ratio_, or an _equilibrium constant_ that controls the variety - quality tradeoff, and also proposes a convergence measure using it.
- [x] **CycleGAN** (2017.5) [[paper](https://arxiv.org/abs/1703.10593)]
    - [x] **DiscoGAN** (2017.5) [[paper](https://arxiv.org/abs/1703.05192)]
    - DiscoGAN and CycleGAN proposes the EXACT SAME learning techniques for style transfer task using GAN, developed independently at the same time.
- [ ] **Frechet Inception Distance (FID)** (2017.6) [[paper](https://arxiv.org/abs/1706.08500)]
    - Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"
- [ ] **ProGAN** (2017.10) [[paper](https://arxiv.org/abs/1710.10196)]
    - Karras et al. "Progressive Growing of GANs for Improved Quality, Stability, and Variation"
- [ ] **PacGAN** (2017.12) [[paper](https://arxiv.org/abs/1712.04086)]
    - Higgins et al. "PacGAN: The power of two samples in generative adversarial networks"
- [ ] **BigGAN** (2018) [[paper](https://arxiv.org/abs/1809.11096)]
- [ ] **GauGAN** (2019.3) [[paper](https://arxiv.org/abs/1903.07291)]
    - Park et al. "Semantic Image Synthesis with Spatially-Adaptive Normalization"
    

### Advanced GANs
- [ ] **DRAGAN** (2017.5) [[paper](https://arxiv.org/abs/1705.07215)]
    - Kodali et al. "On Convergence and Stability of GANs"
- [ ] **Are GANs Created Equal?** (2017.11) [[paper](https://arxiv.org/abs/1711.10337)]
    - Lucic et al. "Are GANs Created Equal? A Large-Scale Study"
- [ ] **SGAN** (2017.12) [[paper](https://arxiv.org/abs/1712.02330)]
    - Chavdarova et al. "SGAN: An Alternative Training of Generative Adversarial Networks"
- [ ] **MaskGAN** (2018.1) [[paper](https://arxiv.org/abs/1801.07736)]
    - Fedus et al. "MaskGAN: Better Text Generation via Filling in the _____"
- [ ] **Spectral Normalization** (2018.2) [[paper](https://arxiv.org/abs/1802.05957)]
    - Miyato et al. "Spectral Normalization for Generative Adversarial Networks"
- [ ] **SAGAN** (2018.5)  [[paper](https://arxiv.org/abs/1805.08318)]  [[tensorflow](https://github.com/brain-research/self-attention-gan)]
    - Zhang et al. "Self-Attention Generative Adversarial Networks"
- [ ] **Unusual Effectiveness of Averaging in GAN Training** (2018) [[paper](https://arxiv.org/abs/1806.04498)]
    - "Benefitting from training on past snapshots."
    - Uses exponential moving averaging (EMA)
- [ ] **Disconnected Manifold Learning** (2018.6) [[paper](https://arxiv.org/abs/1806.00880)]
    - Khayatkhoei, et al. "Disconnected Manifold Learning for Generative Adversarial Networks"
- [ ] **A Note on the Inception Score** (2018.6) [[paper](https://arxiv.org/abs/1801.01973)]
    - Barratt et al., "A Note on the Inception Score"
- [ ] **Which Training Methods for GAN do actually converge?** (2018.7) [[paper](https://arxiv.org/abs/1801.04406)]
    - Mescheder et al., "Which Training Methods for GANs do actually Converge?"
- [ ] **GAN Dissection** (2018.11) [[paper](https://arxiv.org/abs/1811.10597)]
    - Bau et al. "GAN Dissection: Visualizing and Understanding Generative Adversarial Networks"
- [ ] **Improving Generalization and Stability for GANs** (2019.2) [[paper](https://arxiv.org/abs/1902.03984)]
    - Thanh-Tung et al., "Improving Generalization and Stability of Generative Adversarial Networks"
- [ ] Augustus Odena - _"Open Questions about GANs"_ (2019.4) [[distill.pub](https://distill.pub/2019/gan-open-problems/)]
    - Very nice article about current state of GAN research and discusses problems yet to be solved.

### Autoencoders
- [ ] Original autoencoder (1986) [[paper](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_06/PDPVolIChapter8.pdf)]
    - Rumelhart, Hinton, and Williams, "Learning Internal Representations by Error Propagation"
- [x] **AutoEncoder** [[science](https://www.cs.toronto.edu/~hinton/science.pdf)]
    - Hinton et al., "Reducing the Dimensionality of Data with Neural Networks"
- [ ] **Denoising Autoencoders** (2008) [[paper](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)]
    - Vincent et al. "Extracting and Composing Robust Features with Denoising Autoencoders"
- [ ] **Wasserstein Autoencoder** (2017) [[paper](https://arxiv.org/abs/1711.01558)]
    - Tolstikhin et al. "Wasserstein Auto Encoders"

### Autoregressive models
- [ ] **PixelCNN** (2016) [[paper](https://arxiv.org/abs/1606.05328)]
    - van den Oord et al. "Conditional image generation with PixelCNN decoders."
- [ ] **WaveNet** (2016) [[paper](https://arxiv.org/abs/1609.03499)]
    - van den Oord et al. "WaveNet: A Generative Model for Raw Audio"
- [ ] tacotron?


### Layer Normalizations
- [x] **Batch Normalization** (2015.2) [[paper](https://arxiv.org/abs/1502.03167)]
    - Ioeffe et al. "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
- Group Norm
- [x] **Instance Normalization** (2016.7) [[paper](https://arxiv.org/abs/1607.08022)]
    - Ulyanov et al. "Instance Normalization: The Missing Ingredient for Fast Stylization"
- [ ] Santurkar et al. **"How does Batch Normalization help Optimization?"** (2018.5) [[paper](https://arxiv.org/abs/1805.11604)]
- [ ] **Switchable Normalization** (2019) [[paper](https://arxiv.org/abs/1806.10779)]
    - Luo et al. "Differentiable Learning-to-Normalize via Switchable Normalization"
- [ ] **Weight Standardization** (2019.3) [[paper](https://arxiv.org/abs/1903.10520)]
    - Qiao et al. "Weight Standardization"
    
### Initializations
- [ ] **Xavier Initialization** (2010) [[paper](http://proceedings.mlr.press/v9/glorot10a.html)]
    - Glorot et al., "Understanding the difficulty of training deep feedforward neural networks"
- [ ] **Kaiming (He) Initialization** (2015.2) [[paper](https://arxiv.org/abs/1502.01852)]
    - He et al., "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
- [ ] **All you need is a good init** (2015.11) [[paper](https://arxiv.org/abs/1511.06422)]
    - Mishkin et al., "All you need is a good init"
- [ ] **All you need is beyond a good init** (2017.4) [[paper](https://arxiv.org/abs/1703.01827)]
    - Xie et al. "All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation"

### Dropouts
- **Dropout** (2014) [[paper](http://jmlr.org/papers/v15/srivastava14a.html)]
    - Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- **Inverted Dropouts** [[notes on CS231n](http://cs231n.github.io/neural-networks-2/#reg)]
    - Multiplying the inverted _keep_prob_ value on training so that values during inference (or testing) is consistent.
- [ ] Li et al., **"Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift"** (2018.1) [[paper](https://arxiv.org/abs/1801.05134)]


### Meta-Learning / Representation Learning (Zero-Shot learning, Few-Shot learning)
- [x] **Zero-Data Learning** (2008) [[paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-103.pdf)]
    - Larochelle et al., "Zero-data Learning of New Tasks"
- [ ] Palatucci et al., **"Zero-shot Learning with Semantic Output Codes"** (NIPS 2009) [[paper](https://papers.nips.cc/paper/3650-zero-shot-learning-with-semantic-output-codes)]
- [ ] Lampert et al., **"Attribute-Based Classification for Zero-Shot Visual Object Categorization"** (2013.7) [[paper](https://ieeexplore.ieee.org/abstract/document/6571196)]
- [ ] Dinu et al., **"Improving zero-shot learning by mitigating the hubness problem"** (2014.12) [[paper](https://arxiv.org/abs/1412.6568)]
- [ ] Romera-Paredes et al. - **"An embarrassingly simple approach to zero-shot learning"** (2015) [[paper](http://proceedings.mlr.press/v37/romera-paredes15.pdf)]
- [ ] **Prototypical Networks** (2017.3) [[paper](https://arxiv.org/abs/1703.05175)]
    - Snell et al., "Prototypical Networks for Few-shot Learning"
- [ ] **Zero-shot learning - the Good, the Bad and the Ugly"** (2017.3) [[paper](https://arxiv.org/abs/1703.04394)]
    - Xian et al., "Zero-Shot Learning - The Good, the Bad and the Ugly"
- [ ] **Few-Shot learning Survey** (2019.4) [[paper](https://arxiv.org/abs/1904.05046)]
    - Wang et al. "Few-shot Learning: A Survey"


### Transfer learning
- [ ] **Survey 2018** (2018) [[paper](https://arxiv.org/abs/1808.01974)]
    - Tan et al. "A Survey on Deep Transfer Learning"

### Geometric learning
- **Geometric Deep Learning** (2016) [[paper](https://arxiv.org/abs/1611.08097)]
    - Bronstein et al. "Geometric deep learning: going beyond Euclidean data"

### Variational Autoencoders (VAE)
- [ ] **VQ-VAE** (2017.11) [[paper](https://arxiv.org/abs/1711.00937)]
    - van den Oord et al., "Neural Discrete Representation Learning"
- [ ] **Semi-Amortized Variational Autoencoders** (2018.2) [[paper](https://arxiv.org/abs/1802.02550)]
    - Kim et al. "Semi-Amortized Variational Autoencoders"

### Object detection
- [ ] RCNN: https://arxiv.org/abs/1311.2524
- [ ] Fast-RCNN: https://arxiv.org/abs/1504.08083
- [ ] Faster-RCNN: https://arxiv.org/abs/1506.01497
- [ ] SSD: https://arxiv.org/abs/1512.02325
- [ ] YOLO: https://arxiv.org/abs/1506.02640
- [ ] YOLO9000: https://arxiv.org/abs/1612.08242

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
- [ ] **Seq2Seq** (2014) [[paper](https://arxiv.org/abs/1409.3215)]
    - Sutskever et al. "Sequence to sequence learning with neural networks."

### Neural Turing Machine
- [ ] **Neural Turing Machines** (2014) [[paper](https://arxiv.org/abs/1410.5401)]
    - Graves et al., "Neural turing machines."
- [ ] **Pointer Networks** (2015) [[paper]](https://arxiv.org/abs/1506.03134)]
    - Vinyals et al., "Pointer networks."

### Attention / Question-Answering
- [ ] **NMT (Neural Machine Translation)** (2014) [[paper](https://arxiv.org/abs/1409.0473)]
    - Bahdanau et al, "Neural Machine Translation by Jointly Learning to Align and Translate"
- [ ] **Stanford Attentive Reader** (2016.6) [[paper](https://arxiv.org/abs/1606.02858)]
    - Chen et al. "A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task"
- [ ] **BiDAF** (2016.11) [[paper](https://arxiv.org/abs/1611.01603)]
    - Seo et al. "Bidirectional Attention Flow for Machine Comprehension"
- [ ] **DrQA** or **Stanford Attentive Reader++** (2017.3) [[paper](https://arxiv.org/abs/1704.00051)]
    - Chen et al. "Reading Wikipedia to Answer Open-Domain Questions"
- [ ] **Transformer** (2017.8) [[paper](https://arxiv.org/abs/1706.03762)] [[google ai blog](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)]
     -  Vaswani et al. "Attention is all you need"    
- [ ] [read] Lilian Weng - **"Attention? Attention!"** (2018) [[blog_post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)]
    - A nice explanation of attention mechanism and its concepts.
- [ ] **BERT** (2018.10) [[paper](https://arxiv.org/abs/1810.04805)]
    - Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- [ ] **GPT-2** (2019) [[paper (pdf)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)]
    - Radford et al. "Language Models are Unsupervised Multitask Learners"

### Advanced RNNs
- Unitary evolution RNNs : https://arxiv.org/abs/1511.06464
- Recurrent Batch Norm : https://arxiv.org/abs/1603.09025
- Zoneout : https://arxiv.org/abs/1606.01305
- IndRNN : https://arxiv.org/abs/1803.04831
- DilatedRNNs : https://arxiv.org/abs/1710.02224

### Compression
- **MobileNet** (2016) (see above: [Basic CNN Architectures](https://github.com/deNsuh/deep-learning-roadmap/blob/master/README.md#basic-cnn-architectures))
- [ ] **ShuffleNet** (2017)
    - Zhang et al. "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices"

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
- [ ] Unsupervised Representation Learning by Predicting Image Rotations https://arxiv.org/abs/1803.07728
- [ ] Mahjourian et al., **"Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints"** (2018.2) [[paper](https://arxiv.org/abs/1802.05522)]
- [ ] Gordon et al., **"Depth from Videos in the Wild: Unsupervised Monocular Depth Learning from Unknown Cameras"** (2019.4) [[paper](https://arxiv.org/abs/1904.04998)]


### Interpretation and Theory on Generalization, Overfitting, and Learning Capacity
- [ ] **MDL (Minimum Description Length)**
    - Peter Grunwald - "A tutorial introduction to the minimum description length principle" (2004) [[paper](https://arxiv.org/abs/math/0406077)]
- [ ] Grunwald et al., - **"Shannon Information and Kolmogorov Complexity"** (2010) [[paper](https://arxiv.org/abs/cs/0410002)]
- [ ] Dauphin et al. **"Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"** (2014.6) [[paper](https://arxiv.org/abs/1406.2572)]
- [ ] Choromanska et al. **"The Loss Surfaces of Multilayer Networks"** (2014.11) [[paper](https://arxiv.org/abs/1412.0233)]
    - argues that non-convexity in NNs are not a huge problem
- [ ] **Knowledge Distillation** (2015.3) [[paper](https://arxiv.org/abs/1503.02531)]
    - Hinton et al., "Distilling the Knowledge in a Neural Network"
- [ ] **3-Part Learning Theory** by Mostafa Samir
    - [part 1: Introduction](https://mostafa-samir.github.io/ml-theory-pt1/)
    - [part 2: Generalization Bounds](https://mostafa-samir.github.io/ml-theory-pt2/)
    - [part 3: Regularization and Variance-Bias Tradeoff](https://mostafa-samir.github.io/ml-theory-pt3/)
- [ ] **Deconvolution and Checkerboard Artifacts** - Odena (2016) [[distill.pub article](https://distill.pub/2016/deconv-checkerboard/)]
- [ ] Keskar et al. "**On Large-Batch Training for Deep Learning**: Generalization Gap and Sharp Minima" (2016.9) [[paper](https://arxiv.org/abs/1609.04836)]
- [ ] **Rethinking Generalization** (2016.11) [[paper](https://arxiv.org/abs/1611.03530)]
    - Zhang et al. "Understanding deep learning requires rethinking generalization"
- [ ] **Information Bottleneck** (2017) [[paper](https://arxiv.org/abs/1703.00810)] [[original paper on information bottleneck (2000)](https://arxiv.org/abs/physics/0004057)] [[youtube-talk](https://youtu.be/bLqJHjXihK8)] [[article in quantamagazine](https://www.quantamagazine.org/new-theory-cracks-open-the-black-box-of-deep-learning-20170921/)]
    - Shwartz-Ziv and Tishby, "Opening the Black Box of Deep Neural Networks via Information"
- [ ] Neyshabur et al, "Exploring Generalization in Deep Learning" (2017.7) [[paper](https://arxiv.org/abs/1706.08947)]
- [ ] Sun et al., **"Revisiting Unreasonable Effectiveness of Data in Deep Learning Era"** (2017.7) [[paper](https://arxiv.org/abs/1707.02968)]
- [ ] **Super-Convergence** (2017.8) [[paper](https://arxiv.org/abs/1708.07120)]
    - Smith et al. - "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- [ ] **Don't Decay the Learning Rate, Increase the Batch Size** (2017.11) [[paper](https://arxiv.org/abs/1711.00489)]
    - Smith et al. "Don't Decay the Learning Rate, Increase the Batch Size"
- [ ] Hestness et al. **"Deep Learning Scaling is Predictable, Empirically"** (2017.12) [[paper](https://arxiv.org/abs/1712.00409)]
- [ ] **Visualizing loss landscape of neural nets** (2018) [[paper](https://arxiv.org/abs/1712.09913v3)]
- [ ] Olson et al., **"Modern Neural Networks Generalize on Small Data Sets"** (NeurIPS 2018) [[paper](https://papers.nips.cc/paper/7620-modern-neural-networks-generalize-on-small-data-sets)]
- [ ] **Intrinsic Dimension** (2018.4) [[paper](https://arxiv.org/abs/1804.08838)]
    - Li et al., "Measuring the Intrinsic Dimension of Objective Landscapes"
- [ ] Geirhos et al. "**ImageNet-trained CNNs are biased towards texture;** increasing shape bias improves accuracy and robustness" (2018.11) [[paper](https://arxiv.org/abs/1811.12231)]
- [ ] Belkin et al. **"Reconciling modern machine learning and the bias-variance trade-off"** (2018.12) [[paper](https://arxiv.org/abs/1812.11118)]
- [ ] Graetz - "How to visualize convolution features in 40 lines of code" (2019) [[medium](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)]
- [ ] Geiger et al. **"Scaling description of generalization with number of parameters in deep learning"** (2019.1) [[paper](https://arxiv.org/abs/1901.01608)]
- [ ] **Are all layers created equal?** (2019.2) [[paper](https://arxiv.org/abs/1902.01996)]
    - Zhang et al. "Are all layers created equal?" 
- [x] Lilian Weng - **"Are Deep Neural Networks Dramatically Overfitted?"** (2019.4) [[lil'log](https://lilianweng.github.io/lil-log/2019/03/14/are-deep-neural-networks-dramatically-overfitted.html)]
    - Excellent article about generalization and overfitting of deep neural networks
    
### Adversarial Attacks and Defense against attacks (RobustML)
- [RobustML site](https://www.robust-ml.org/)
- [x] **Adversarial Examples** Szegedy et al. - Intreguing Properties of Neural Networks (2013.12) [[paper](https://arxiv.org/abs/1312.6199)]
    - induces missclassification by applying small perturbations
    - this paper was the first to coin the term "Adversarial Example"
- [ ] **Fast Gradient Sign Attack (FGSM)** (2014.12)
    - Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015) [[paper](https://arxiv.org/abs/1412.6572)]
    - This paper presented the famous "panda example" (as also seen in [pytorch tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html))
- [ ] Kurakin et al., **"Adversarial Machine Learning at Scale"** (2016.11) [[paper](https://arxiv.org/abs/1611.01236)]
- [ ] Mandry et al., **"Towards Deep Learning Models Resistant to Adversarial Attacks"** (2017.6) [[paper](https://arxiv.org/abs/1706.06083)]
- [ ] Carlini et al., **"Audio Adversarial Examples: Targeted Attacks on Speech-to-Text"** (2018.1) [[paper](https://arxiv.org/abs/1801.01944)]


### Neural architecture search (NAS) and AutoML
- **GREAT AutoML Website** [[site](https://www.automl.org/)]
    - They maintain a blog, a list of NAS literatures, analysis page, and a web book.
- [ ] **AdaNet** (2016.7) [[paper](https://arxiv.org/abs/1607.01097?context=cs)] [[GoogleAI blog](https://ai.googleblog.com/2018/10/introducing-adanet-fast-and-flexible.html)]
    - Cortes et al. "AdaNet: Adaptive Structural Learning of Artificial Neural Networks"
- [ ] **NAS** (2016.12) [[paper](https://arxiv.org/abs/1611.01578)]
    - Zoph et al. "Neural Architecture Search with Reinforcement Learning"
- [ ] **PNAS** (2017.12) [[paper](https://arxiv.org/abs/1712.00559)]
    - Liu et al. "Progressive Neural Architecture Search"
- [ ] **ENAS** (2018.2) [[paper](https://arxiv.org/abs/1802.03268)]
    - Pham et al. "Efficient Neural Architecture Search via Parameter Sharing"
- [ ] **NAS Survey 2018** (2018.4) [[paper](https://arxiv.org/abs/1808.05377)]
- [ ] **DARTS** (2018.6) [[paper](https://arxiv.org/abs/1806.09055)]
    - Liu et al. "DARTS: Differentiable Architecture Search"
    - Uses a continuous relaxation over the discrete neural architecture space.
- [ ] **RandWire** (2019) [[paper](https://arxiv.org/abs/1904.01569)]
    - Xie et al. "Exploring Randomly Wired Neural Networks for Image Recognition" [Facebook AI Research]
    
    
### Practical Techniques
- [x] Andrej Karpathy - **"A recipe for training neural networks"** (2019) [[Andrej Karpathy Blog Post](http://karpathy.github.io/2019/04/25/recipe/)]
    

## DL roadmap reference
- https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap
- https://github.com/terryum/awesome-deep-learning-papers
- which DL algorithms should I implement to learn?
https://www.reddit.com/r/MachineLearning/comments/8vmuet/d_what_deep_learning_papers_should_i_implement_to/


### Theory
- [The MML(Mathematics for Machine Learning) book](https://mml-book.github.io/)
- [Andrej Karpathy - Yes You Should Understand Backprop](https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b)
- [Theoretical principles for Deep Learning](http://mitliagkas.github.io/ift6085-dl-theory-class-2019/?fbclid=IwAR02mfw0hMM5UFaBuxBgDi5wfT6TaIt35wktZGpCmmhu0_3GMA6HqFN1GFs)
- [Stanford STATS 385 - Theories of Deep Learning](https://stats385.github.io/)
    - [Lecture Videos](https://www.researchgate.net/project/Theories-of-Deep-Learning?fbclid=IwAR0dwnuswA1jMwIOuydb_a83AM22FfuD6PpAWPiIW-76OCemcRBrVVLKLoM)
- CSC 231 notes : http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/


### Resources
- **A Selective Overview of Deep Learning** (2019) [[paper](https://arxiv.org/abs/1904.05526)]
    - Fan et al. "A Selective Overview of Deep Learning"
    - A nice overview paper on deep learning up to early 2019 (about 30 pages)
