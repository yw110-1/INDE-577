# Datasets
The datasets in the whole repository that I will use are showing as below:

## Palmer Penguin Dataset
The penguin data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network. The dataset contains 343 rows and 7 columns. The variables of the dataset are:
  - ```species```: penguin species (Chinstrap, Adélie, or Gentoo)
  - ```bill_length_mm```: culmen length (mm)
  - ```bill_depth_mm```: culmen depth (mm)
  - ```flipper_length_mm```: flipper length (mm)
  - ```body_mass_g```: body mass (g)
  - ```island```: island name (Dream, Torgersen, or Biscoe) in the Palmer Archipelago (Antarctica)
  - ```sex```: penguin sex
  
## Abalone Dataset
The abalone data comes from an original (non-machine-learning) study: Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994), "The Population Biology of Abalone (_Haliotis_ species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the North Coast and Islands of Bass Strait", Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288). The variables of data includes:
  - ```Sex```: Male (M), Female (F), or Infant (I)
  - ```Length```: Longest shell measurement
  - ```Diameter ```: Perpendicular to length
  - ```Height```: Height with meat in shell
  - ```Whole weight```: Weight with whole abalone
  - ```Shucked weight```: Weight of meat
  - ```Viscera weight```: Gut weight (after bleeding)
  - ```Shell weight```: Weight after being dried
  - ```Rings```: +1.5 gives the age in years

## Load_wine Dataset
The make_classification dataset is from sklearn.datasets package. It can be used for classification problems. It contains three classes (types of wine), and it contains 13 attributes:
 - ```Alcohol```
 - ```Malic acid```
 - ```Ash```
 -  ```Alcalinity of ash```  
 - ```Magnesium```
 - ```Total phenols```
 - ```Flavanoids```
 - ```Nonflavanoid phenols```
 - ```Proanthocyanins```
 - ```Color intensity```
 - ```Hue```
 - ```OD280/OD315 of diluted wines```
 - ```Proline```
 
## Fashion MNIST Dataset
Fashion-MNIST is a dataset of Zalando's article images consisting of a training set of 60,000 examples and a test set of 10,000 examples. It can be an alternative to MNIST dataset. Each example is a 28x28 grayscale image, associated with a label from 10 classes: ```Ankle boot```,  ```Bag```, ```Coat```,  ```Dress```,```Pullover```,```Sandal```,```Shirt```,```Sneaker```,```T-shirt/top```, and```Trouser```. The image below shows an example of dataset with ```Bag``` label.

<p align="center">
<img src="https://github.com/yw110-1/INDE-577/blob/main/Supervised%20Learning/Perceptron/image/bag.png" alt="bag" width="700"/>
</p>

## Make_classification Dataset
The make_classification dataset is from sklearn.datasets package. It can be used for classification problems similar to make_blobs dataset.

## Reference
1. Palmer Penguin Dataset: https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data
2. Make_classification Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
3. Abalone Dataset: http://archive.ics.uci.edu/ml/datasets/Abalone
4. Load_wine Dataset: https://archive.ics.uci.edu/ml/datasets/wine
5. Fashion_MNIST Dataset: https://www.tensorflow.org/datasets/catalog/fashion_mnist
