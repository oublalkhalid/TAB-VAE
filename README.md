# TAB-VAE: Temporal Attention Bottleneck for VAE is informative? 

ICML 2023 Workshop on DeployGenerativeModel: Temporal Attention Bottleneck for VAE is informative? 

The use of generative models in energy disaggregation has attracted attention to address the challenge of source separation. This approach holds promise for promoting energy conservation by enabling homeowners to obtain detailed information on their energy consumption solely through the analysis of aggregated load curves. Nevertheless, the model's ability to generalize and its interpretability remain two major challenges. To tackle these challenges, we deploy a generative model called TAB-VAE (Temporal Attention Bottleneck for Variational Autoencoder), based on hierarchical architecture, addresses signature variability, and provides a robust, interpretable separation through the design of its informative representation of latent space.

<img src="docs/img/overview.png" alt="overview" width="100%" height="100%"/>

## Temporal Attention Bottleneck Cell

<img src="docs/img/Temporal_attention_cell.gif" alt="drawing" width="50%" height="50%"/>


## Run Experiment

In order to execute the experiment run:

- Install requirements 

```
pip install -r requirements.txt
```

```python
python main.py --root_path /TAB-VAR --data_path /dataset/Uk-dale --input_dim 3 --beta_end 0.1
```

## Datasets
The **NILMTK**[2] toolkit is used for reading the data.
All the datasets that are compatible with **NILMTK** are supported, but the benchmark
is constructed on end-uses from **UK DALE**[3], **REDD**[4] and **REFIT**[5]. 
It should be noted that the data have to be downloaded manually.
In order to load the data, the files _path_manager.py_ and _datasource.py_ inside _datasources/_ directory should be 
modified accordingly.

## Resultats

<img src="docs/img/results_7device.png" alt="results_7device" width="100%" height="100%"/>


## References
1. Symeonidis, N.; Nalmpantis, C.; Vrakas, D. A Benchmark Framework to Evaluate Energy Disaggregation Solutions. International 541
Conference on Engineering Applications of Neural Networks. Springer, 2019, pp. 19–30.
2. Batra, N.; Kelly, J.; Parson, O.; Dutta, H.; Knottenbelt, W.; Rogers, A.; Singh, A.; Srivastava, M. NILMTK: an open source toolkit 525
for non-intrusive load monitoring. Proceedings of the 5th international conference on Future energy systems, 2014, pp. 265–276.
3. Jack, K.; William, K. The UK-DALE dataset domestic appliance-level electricity demand and whole-house demand from five UK
homes. Sci. Data 2015, 2, 150007.
4. Kolter, J.Z.; Johnson, M.J. REDD: A public data set for energy disaggregation research. Workshop on data mining applications in
sustainability (SIGKDD), San Diego, CA, 2011, Vol. 25, pp. 59–62.
5. Firth, S.; Kane, T.; Dimitriou, V.; Hassan, T.; Fouchal, F.; Coleman, M.; Webb, L. REFIT Smart Home dataset, 2017.
doi:10.17028/rd.lboro.2070091.v1.

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

