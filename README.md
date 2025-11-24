# Use of autoencoders for the modeling of term structure of interest rates.

This project aims to model the term structure of interest rates using Autoencoders (AE) and Variational Autoencoder (VAE) networks, and to investigate the forward rate dynamics under _Q_ and _P_ measures. Most folders contain various Python scripts and the output/generated `.csv` and `.png` files.

## Chapter 2

The relevant folders of Chapter 2 are `Data`, `parametric_models` and `autoencoders_networks` in sequence of reading.

- `Data`

Contains the five currencies yield datasets of this Chapter where each currency has 400 yield curves.

- `parametric_models`

Contains the calibration functions of the Nelson-Siegel(-Svensson) interest rate curve models. There are various `yplot` scripts that implement pointwise and grid-search calibrations and plot the yield curves.

- `autoencoders_networks`

Contains the AE and VAE networks, the pre-training of the networks using synthetic Svensson curves (each with respect to its own currency) and the plotting of yield curves in `yplot` scripts.

## Chapter 3

The relevant folders of Chapter 3 are `data`, `pca`, `vae` and `statistical_evaluation` in sequence of reading.

- `data`

Contains the instantaneous forward rates of the GBP with 1264 forward curves.

- `pca`

Contains the Principal Component Analysis (PCA) functions, historical forward plot and simulation of rates using Heath-Jarrow-Morton (HJM) Stochastic Differential Equation (SDE).

- `vae`

Contains the script to fit the VAE network on the forward dataset whereby the network is trained on mini-batch size of 64. Thereafter, we simulate the forward rates using HJM SDE from the generated VAE latent factors.

- `statistical_evaluation`

Contains the Jarque-Bera test and Predictive Root-Mean-Squared-Error (RMSE) functions to derive statistical inference of the simulated forward rates from PCA and VAE methodologies.

## Chapter 4

The relevant folders of Chapter 4 are `data`, `pca` and `statistical_evaluation` in sequence of reading.

- `data`

Contains the instantaneous forward rates, short rates of the Great British Pounds (GBP) and the discretized volatility of the PCA methodology derived from _Q_ measure in Chapter 3.

- `pca`

Contains the simulation of rates using Hull-Sokol-White (HSW) SDE.

- `statistical_evaluation`

Contains the Jarque-Bera test and predictive RMSE functions to derive statistical inference of the simulated forward rates under _P_ measure.

## Chapter 5

The relevant folders of Chapter 5 are `bloomberg_data`, `swap_rate_computation` and `implied_volatility_computation` in sequence of reading.

- `bloomberg_data`

Contains the at-the-money LIBOR swaption strike and market price for swaptions with expiries and tenors; 3M x 1Y, 10Y x 5Y, 25Y x 20Y.

- `swap_rate_computation`

Contains the swap rate computation of the aforementioned swaptions using the simulated forward rates from PCA and VAE methodologies under _Q_ measure, taking the first 20 time stamp.

- `implied_volatility_computation`

Contains the script to compute implied volatility of swaption by solving the Black-76 call formula using the Newton-Raphson method.

## Environment

The experiments were conducted on Windows 11 Pro using Python 3.11 in Visual Studio Code. The hardware configuration includes an AMD Ryzen 7 5800X 8-Core Processor (3.80 GHz) and 64 GB RAM.
