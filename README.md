# Research Paper: Neural Network Moon Crater Detection and Age Prediction
Based on a research paper by Yang. et al. in 2020 titled [Lunar impact crater identification and age estimation](https://www.nature.com/articles/s41467-020-20215-y), this hybrid model was created to experiment with a Convolutional + Feedforward architecture to perform lunar impact crater age classification across 6 time periods spanning 4 billion years in time. Utilizing approx. 1700 [crater images](https://data.lroc.im-ldi.com/lroc/view_rdr/WAC_GLOBAL) from the [Lunar Reconnasissance Orbiter camera](https://lroc.im-ldi.com/) (LROC) and corresponding age-labeled [crater metadata](https://www.google.com/url?q=https://www.lpi.usra.edu/lunar/surface/Lunar_Impact_Crater_Database_v08Sep2015.xls&sa=D&source=docs&ust=1748971076400682&usg=AOvVaw3khvuiNDHE97KfINOCzDDx) from the [Lunar and Planetary Institute](https://www.lpi.usra.edu/) (LPI), a maximum test accuracy of 62% was achieved.
> **Authors:** Jonathan Bennett, Evan Cedeno, Adam Miller, and Ferris Wolf

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <ul>
        <li><a href="##Introduction">Introduction</a></li>
        <li><a href="##Research-Paper-Review">Research paper review</a></li>
        <li><a href="##Description-of-Dataset">Description of dataset</a></li>
        <li><a href="##Description-of-Our-Model">Description of our model</a></li>
        <li><a href="##Training-The-Model">Training the model</a></li>
        <li><a href="##Analysis-of-results/Future-endeavors">Analysis of results/Future endeavors</a></li>
      </ul>
    </li>
    <li><a href="#next-steps">Next Steps</a></li>
    <li><a href="#citations">Citations</a></li>
    <li><a href="#contributions">Contributions</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Our moon has no atmosphere or active core, and thus its environment does not change over time. As a result, impact craters from space debris are preserved in its landscape. For billions of years, the moon has been repeatedly hit by asteroids, leaving their mark on its terrain. Craters are the lunar equivalent of fossils, and like archeologists, astronomers try to deduce when these remnants were first produced. While some craters’ time periods can be deduced, most cannot due to lack of evidence. Recently, the rise of Machine Learning has enabled scientists to analyse craters en masse to predict the ages of thousands. One paper in particular used deep neural networks to predict the ages of a large number of craters. This project serves to make use of their strategy to predict ages of craters sampled from a different dataset, working to expand the domain of predicted ages. 

## Research paper review
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Published to Nature in December 2020, Lunar impact crater identification and age estimation with Chang’E data by deep and transfer learning by C. Yang et al. began with data from two orbiters, China’s Chang’E-1 (CE-1) and Chang’E-2 (CE-2). Their goal was to 1) identify craters over 1 kilometer in diameter and 2) predict ages of these craters. They looked at mid to low latitudes of the moon for craters ranging from 1 km to hundreds of kilometers. In order to compare results with known literature, Yang et al. also used recognized crater catalogs, including the International Astronomical Union (IAU) for location and the Lunar and Planetary Institute (LPI) for ages. These catalogs had around 10,000 craters and 1,400 corresponding ages and served as training data for the crater detection model. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The lunar crater detection model employs a two-stage approach. Stage One: A region-based convolutional detector was fine-tuned on CE-1 images (both high- and lower-resolution subsets). This model learned to identify craters from ~1 km up to hundreds of kilometers in diameter. Stage Two: The trained model was then transferred directly (with no further labeled training) to the CE-2 images. Although CE-2 data have different resolution and slightly different appearance, the authors showed that transfer learning preserved most of the detection accuracy for smaller, sharper craters (since CE-2 resolution is finer).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Once around 117,000 craters were identified by this model, they were then fit into a secondary model, aimed at estimating their ages. Choosing an age in years is quite difficult, as it cannot easily be predicted by image alone. Recognizing this, Yang et al. transformed their model into one of classification, instead predicting the time periods based on “morphological and stratigraphic features.” This was yet again a dual-channel model, combining image data and numerical features into a classification model which predicted which time period the crater came from. Training on the 1,400 craters from the age dataset, the researchers claimed to achieve a 85-89% accuracy in assigning a crater to the correct system. They then applied this on the 117,000 craters they had detected to predict their ages, although any reliability of the accuracy of those is impossible without additional verification/research.


## Description of dataset
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A significant portion of the dataset consists of images of 1,675 individual moon craters. To obtain these images, we first compiled the names, radii, latitudes, and longitudes of the craters we were interested in. In addition to this information, we used a dataset of large moon images from the Lunar Reconnaissance Orbiter Camera (LROC), that together comprise the entire surface of the moon. In this moon image dataset, there are ten images. Eight of these images are rectangular projections that in total span from -60 to 60 degrees latitude and from 0 to 360 degrees longitude. The other 2 images are polar projections of the north and south poles of the moon which each have 360 degrees of longitude and span from -60 to -90 degrees latitude and 60 to 90 degrees latitude. Since the latitudes and longitudes of each crater in the dataset is known, we were able to find its corresponding moon image. After this was done, each crater’s latitude and longitude was mapped to its corresponding image and a new image was created of that specific location. Each crater image is slightly larger than the crater itself so that no details are lost. However, since a CNN requires all images to be the same size, the images are all resized to 512x512 pixels.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The other half of the dataset consists of known parameters on lunar craters (e.g. Radius, Depth) that were compiled by the Lunar and Planetary Institute (LPI) into the Lunar Impact Crater Database (LICD). Of the 8717 craters depicted in the dataset, we extracted all 1,675 craters with identified ages that span the six time periods (pre-Nectarian, Nectarian, Lower Imbrian, Upper Imbrian, Eratosthenian, and Copernican). Additionally, we extracted extra parameters including crater radius and depth to supply our feed forward neural network. 

## Description of our model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Our model follows the general architecture of Yang. et al.. We have a Convolutional Neural Network (CNN) and a FeedForward Network (FFN) that concatenate results into a Classification model, which predicts the time period of the crater. We begin with the Convolutional Neural Network. As previously mentioned, our CNN resizes all images to 512x512 pixels. From there, it applies both linear and ReLU non-linear transformations into a one dimensional array of size 128. Our Feedforward Network makes use of our numerical parameters, again applying linear and non-linear transformations to expand the impact of these values in our model. Once both networks have reshaped their respective data, the results are then concatenated into a single array. This final array is sent to the classification neural network, which applies one last set of transformations. Finally, the model uses all data to predict the age classification of each crater. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Given that our dataset comprises roughly 1,600 craters, (which is similar to the size of Yang. et al.), we have the opportunity to train on our whole dataset en masse. Our training consists of 100 epochs. After each epoch, we evaluate the current state of our model for its train accuracy and test accuracy. Our model is then scored based on these results, the current iteration is saved, and it moves on to the next epoch. After training, we look at the accuracies over all epochs and keep the model with the best score. 

## Training the model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To evaluate the hybrid model before scaling to full training, we varied the model's hyperparameters by performing 1 epoch mini trials. As shown in Table 1 (below), we varied the dropout rate, learning rate, hidden layer sizes, and feed forward parameters to land on a few models that performed the best. Although these results would likely vary drastically for full training, these tests helped us narrow down our hyperparameters choices to a few models that we were confident in to focus our resources towards for the upcoming trials. Additional tests were also done with 5 and 10 epochs on the most successful hyperparameter choices.

<img src="https://github.com/EvanC8/Lunar-Crater-Age-Classification/blob/main/performance_images/model-fitting.png?raw=true" height="150">

> **Table 1:** Results for the 1-epoch mini trials. Values changed were parameters, dropout rate, and learning rate. Values recorded were loss function, accuracy of training data, and accuracy of test data. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For instance, we noticed that when adding more crater parameters to the feedforward design, the train and test accuracies fell significantly. We suspect that this was due to two reasons: Firstly, including additional parameters led the hybrid model to overtrain on the feedforward side, paying less significance to the CNN output. Secondly, we found that many of the metadata stored on each crater were redundant and even proportional to an extent, like the depth of a crater compared to its cavity height. Thus, we found that restricting the feed forward network to one or two parameters like radius and depth were sufficient to train our hybrid model without overfitting on redundant, potentially unnecessary feedforward parameters.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Once the desired hyperparameters were chosen, we were ready to begin training. Our data was randomly split 80/20 into training and test data, and we performed 100 epochs on our model. After each epoch, we saved the current model and evaluated for accuracy on both training and test data. Because of our limited dataset, our model began to overfit around 40 epochs, so we stopped training after that point. Our best model was able to perform with an accuracy of 62% on our test data, see Figure 1 (Below).

<img src="https://github.com/EvanC8/Lunar-Crater-Age-Classification/blob/main/performance_images/hybrid_best.png?raw=true" width="500">

> **Figure 1:** Results from best hybrid model. The model begins to overtrain around epoch 40, causing a downward trend in test accuracy. The best model had a test accuracy of 62%. 

## Analysis of results/Future endeavors
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The limitations of our model were largely due to our dataset. This caps how long we can train it for. Our biggest issue was the implementation of dropout, allowing the model to train further without worry of overfitting. Upon using non-zero dropout rates, the model would converge to predicting Nectarian for every crater. We believe that the significant imbalance between crater age categories contributed to this result. This issue was eventually resolved, delaying overfitting of our model by around 20 epochs or more in most cases. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Furthermore, we wished to compare the accuracy of our models with larger changes. Specifically, compared the CNN image classifier by itself, without numerical attributes from the FFN. This model was able to work with dropout, producing our highest test accuracy of 63.5%, see Figure 2 (below). This would suggest that the chosen numerical attributes, plugged into our Feedforward network, hinder the ability of our classifier to accurately predict crater ages. 

<img src="https://github.com/EvanC8/Lunar-Crater-Age-Classification/blob/main/performance_images/cnn_best.png?raw=true" width="500">

> **Figure 2:** Results from the CNN into Classifier model, without input from the FFN.. This non-hybrid model produces our best test accuracy of 63.5%, without discernible overtraining. 
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;When we tested the system using only the FFN, we found that regardless of hyperparameters used, the system would only ever output every crater as being from class 1 (Nectarian, the most common class). This suggests that the data we are feeding the system via the FFN is negligible in providing any information about the crater’s age, as does the increased accuracy of the CNN-only model. Scientifically this is likely due to the fact that impactors will cause similarly quantitative values for craters regardless of age over a certain size. The CNN is effective in that it looks for the qualitative effects that erosion-like processes and ejecta covering would have on craters over billions of years. As for why Yang et al.’s FFN-hybrid model was effective, it likely has to do with the size of craters we used. Their dataset included many smaller craters which fade at a faster rate than larger ones. Since our dataset is composed of only relatively large craters, that factor is not useful for our identification.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We recognize that our current, best model is proficient in its reduced form, without input from our Feedforward network. Whether it’s more epochs to train, optimizing hyper parameters, or choosing which numerical attributes to give the FFN, we predict that an optimized hybrid model would improve the model’s ability to accurately identify crater ages, including input from numerical crater attributes. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The next steps for our model would be to predict the ages of more craters. Most of the crater ages in the Lunar Impact Crater Database are unlabeled. This means that the rest of the craters can be immediately plugged into our model. However, this falls under the assumption that our model has everything it needs to label the ages, even if the datasets aren’t identically distributed. 


# Citations
* Research paper reference: [Lunar impact crater identification and age estimation](https://www.nature.com/articles/s41467-020-20215-y)
* Crater image data source: [Lunar Reconnasissance Orbiter camera](https://lroc.im-ldi.com/)
* Crater metadata and age labels source: [Lunar and Planetary Institute](https://www.lpi.usra.edu/)

# Contributions
* This research project was carried out in collaboration with Jonathan Bennett, Adam Miller, and Ferris Wolf

# Contact
Evan Cedeno - escedeno8@gmail.com

Project Link: [Lunear Impact Crater Decection and Age Prediction](https://github.com/EvanC8/Lunar-Crater-Age-Classification)

