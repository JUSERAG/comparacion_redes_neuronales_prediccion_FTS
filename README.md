# comparacion_redes_neuronales_prediccion_FTS

Este repositorio almacena 3 modelos de redes neronales los cuales fueron utlizados para la predicción
de series temporales financieras (FTS). Los modelos que va a encontrar a continuación son: 
    - Temporal Convolutional Network (TCN).
    - Dendritic Neural Model (DNM).
    - Bayesian Neural Network (BNN).

Los modelos no fueron escritos desde cero, sino que están basados en modelos encontrados en otros
repositorios de GITHUB los cuales fueron adaptados para las necesidad de predecir ciertos tipos de 
activos de renta variable, entre los que se encuentran indices, acciones, divisas, materias primas y criptoactivos.

Los tres modelos se extrajeron de los siguientes repositorios:
 - DNM
    https://github.com/ChengTANG-AI/Scale-Free-Differential-Evolution/tree/main/SFDE_DNM_Code
    https://github.com/google-deepmind/deepmind-research/blob/master/gated_linear_networks/colabs/dendritic_gated_network.ipynb
 - TCN
    https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
 - BNN
    https://github.com/tennisonliu/bayesian-neural-network/blob/master/config.py


La data de los activos se extrajo de la API de yahoo finance.
https://developer.yahoo.com/api/


Iniciar entorno virtual 
- .\Script\activate


Librerias a instalar
- pip install numpy pandas yfinance scikit-learn matplotlib torch 

Los modelos se encuentran en la carpeta /modelos/