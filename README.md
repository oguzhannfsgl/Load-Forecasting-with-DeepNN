# Load-Forecasting-with-DeepNN
Implementation of two different models (TF2/Keras) from literature and a custom model for day-ahead load forecasting (short term load forecasting) on two different datasets.

These two models from literature are [this](https://www.researchgate.net/publication/322547645_A_High_Precision_Artificial_Neural_Networks_Model_for_Short-Term_Energy_Load_Forecasting) which they call 'DeepEnergy' and [this](https://www.researchgate.net/publication/354655798_A_Two-Stage_Short-Term_Load_Forecasting_Method_Using_Long_Short-Term_Memory_and_Multilayer_Perceptron).

### USAGE
#### From linux terminal:
* #### Clone the repo
  ```git clone https://github.com/oguzhannfsgl/Load-Forecasting-with-DeepNN```
* #### Get into the source file
  ```cd Load-Forecasting-with-DeepNN/src```
* #### Install neccessary libraries
  ```pip3 install -r requirements.txt```
* #### Train a model with a dataset
  ```python3 main.py --data_from SPAIN --model_name seqmlp --epochs 30```
