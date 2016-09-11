# TextureClassificationKMeans
#Classifying Textured Areas using Law filters and K-Means segmentation
Level – Wave , Level – Ripple , Edge – Level , Spot – Wave ,Wave -Edge  - (Determined this combination based on a BF method, where I go over all possible combinations – 25C5 – around 53000 , and find the combination of masks that will give me the most balanced quadrants  create a Histogram of the clustered image /or best labels, then find a least squared error of the 4-bin counts. 
#
(binCount1 – 0.25)^2 + (binCount2 – 0.25)^2 + (binCount3 – 0.25)^2 + (binCount4 – 0.25)^2 = sum
#
I am minimizing the sum , and find the combination that gives the min value.


#Input Image:

![Alt text](/1.png?raw=true "Input Image")

#Gray-Level Clustering with NO_CLUSTERS=4 

![Alt text](/2.png?raw=true "NO_CLUSTERS=4")

#Gray-Level Clustering with NO_CLUSTERS=12

![Alt text](/3.png?raw=true "NO_CLUSTERS=12")

#Texture Clustering :
#For the laws energy measure kernels combination:
#NO_CLUSTERS=4

![Alt text](/4.png?raw=true "Law's Energy Measure NO_CLUSTERS=4")
